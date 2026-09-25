"""
run_live_master_pipeline.py  -- interleaved balanced sweep
=========================================================
Usage:
    python run_live_master_pipeline.py [--max_new_per_cell N] [--day LABEL]

Flags:
    --max_new_per_cell N   Stop adding new runs to a cell once it has
                           N runs on disk. Default: 26 (N=26 balanced design).
    --day LABEL            Optional label printed in logs (e.g. "day1").

Scheduling strategy
-------------------
Rather than exhausting one condition before touching the next, the outer
loop interleaves across conditions:

    for pass_idx in 0, 1, 2, ...:
        for model in [qwen, 20b, 120b]:
            for condition in [clean, poisoned_explicit, poisoned_implicit]:
                pick one (task, trial) pair that is still below the cap
                run it

Each pass adds at most one run per (model, condition) pair per iteration,
guaranteeing that all cells grow in parallel.  The run exits when every
non-exhausted cell has reached max_new_per_cell new completions this
session (or when a model hits its daily quota).

End-of-session report
---------------------
At exit, logs:
  - per-cell counts (on disk)
  - planned vs actual new runs this session
  - per-model token consumption and observed refill rate
  - quota events
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from harness.agent_loop import AgentLoop
from harness.groq_client import RateLimitedGroqClient
from harness.scorer import (
    aggregate_experiment_results,
    evaluate_run,
    two_proportion_z_test,
    wilson_ci,
)
from tasks.task_loader import load_tasks


# Force UTF-8 output on Windows (prevents UnicodeEncodeError in print())
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SESSION_START = time.time()
LOG_PATH = "results/live_sweep.log"


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    os.makedirs("results", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Per-model token-refill rate snapshot logger
# ---------------------------------------------------------------------------

def log_refill_snapshot(client: RateLimitedGroqClient, label: str = "") -> None:
    """Print the current per-model token-rate snapshot to log."""
    report = client.get_model_refill_report()
    wall_h = (time.time() - SESSION_START) / 3600.0
    log(f"=== Token-refill snapshot{' [' + label + ']' if label else ''} -- session wall time {wall_h*60:.1f} min ===")
    for model, s in report.items():
        log(
            f"  {model}: consumed={s['tokens_consumed']:,} | calls={s['call_count']} | "
            f"avg_tok/call={s['avg_tokens_per_call']:.0f} | "
            f"avg_tok/h={s['avg_tokens_per_hour']:.0f} | "
            f"est_remaining_tpd={s['est_remaining_tpd']:,} | "
            f"tpd_hits={s['tpd_hits']}"
        )
        if s["tph_samples"]:
            log(f"    tph_samples (every 10 calls): {s['tph_samples']}")


# ---------------------------------------------------------------------------
# ASR / task-completion independence audit (Item 1 re-check at N=26)
# ---------------------------------------------------------------------------

def independence_audit(
    records: List[Dict[str, Any]],
    model: str,
    condition: str,
) -> Dict[str, Any]:
    """
    For a given (model, condition) cell, compute per-row attack_succeeded
    vs task_completed and test for degeneracy.

    Returns a dict with:
      - n_rows
      - both_true, both_false, attack_only, task_only
      - is_degenerate: True if attack_only == 0 and task_only == 0
      - phi_coefficient: Matthews correlation coefficient (±1 = perfect correlation)
    """
    rows = [
        r for r in records
        if r.get("model") == model and r.get("condition") == condition
    ]
    n = len(rows)
    if n == 0:
        return {"model": model, "condition": condition, "n_rows": 0}

    both_true = sum(1 for r in rows if r.get("attack_succeeded") and r.get("task_completed"))
    both_false = sum(1 for r in rows if not r.get("attack_succeeded") and not r.get("task_completed"))
    attack_only = sum(1 for r in rows if r.get("attack_succeeded") and not r.get("task_completed"))
    task_only = sum(1 for r in rows if not r.get("attack_succeeded") and r.get("task_completed"))

    # Matthews correlation coefficient (phi) for 2x2 table
    # Table: TP=both_true, FP=attack_only, FN=task_only, TN=both_false
    tp, fp, fn, tn = both_true, attack_only, task_only, both_false
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    import math
    phi = (tp * tn - fp * fn) / math.sqrt(denom) if denom > 0 else float("nan")

    is_degenerate = (attack_only == 0 and task_only == 0)

    return {
        "model": model,
        "condition": condition,
        "n_rows": n,
        "both_true": both_true,
        "both_false": both_false,
        "attack_only": attack_only,
        "task_only": task_only,
        "is_degenerate": is_degenerate,
        "phi_coefficient": round(phi, 4) if not math.isnan(phi) else None,
    }


def run_independence_audits(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run audit for all (model, condition) pairs present in records."""
    models = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    conditions = ["clean", "poisoned_explicit", "poisoned_implicit"]
    results = []
    for m in models:
        for c in conditions:
            audit = independence_audit(records, m, c)
            if audit.get("n_rows", 0) > 0:
                results.append(audit)
    return results


# ---------------------------------------------------------------------------
# End-of-session report
# ---------------------------------------------------------------------------

def write_end_of_session_report(
    records: List[Dict[str, Any]],
    client: RateLimitedGroqClient,
    new_runs_this_session: int,
    planned_new_per_cell: int,
    day_label: str,
    exhausted_models: Set[str],
) -> None:
    """Write the end-of-session summary to log and to a JSON file."""
    models = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    conditions = ["clean", "poisoned_explicit", "poisoned_implicit"]

    counts = Counter((r.get("model"), r.get("condition")) for r in records)

    log("=" * 70)
    log(f"END-OF-SESSION REPORT [{day_label}]")
    log("=" * 70)
    log(f"New runs completed this session: {new_runs_this_session}")
    log(f"Total runs on disk: {len(records)}")
    log("")
    log("Per-cell counts (on disk):")
    for m in models:
        for c in conditions:
            n = counts.get((m, c), 0)
            flag = " [OK]" if n >= planned_new_per_cell else f" (need {planned_new_per_cell - n} more)"
            log(f"  {m:30s} | {c:20s} | {n:>3}{flag}")

    log("")
    log_refill_snapshot(client, label="end-of-session")

    # Independence audit for all cells with enough data
    log("")
    log("ASR / task-completion independence audit:")
    audits = run_independence_audits(records)
    for a in audits:
        degenerate_str = " [DEGENERATE]" if a["is_degenerate"] else ""
        log(
            f"  {a['model']:30s} | {a['condition']:20s} | N={a['n_rows']:>3} | "
            f"both_T={a['both_true']:>3} both_F={a['both_false']:>3} "
            f"atk_only={a['attack_only']:>2} task_only={a['task_only']:>2} "
            f"phi={a['phi_coefficient']}{degenerate_str}"
        )

    log("")
    log("Exhausted models (daily quota hit):")
    if exhausted_models:
        for m in exhausted_models:
            log(f"  {m}")
    else:
        log("  (none)")

    # Write machine-readable summary
    stats = client.get_execution_stats()
    summary = {
        "day_label": day_label,
        "session_start_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(SESSION_START)
        ),
        "session_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "new_runs_this_session": new_runs_this_session,
        "total_runs_on_disk": len(records),
        "planned_new_per_cell": planned_new_per_cell,
        "per_cell_counts": {
            f"{m}|{c}": counts.get((m, c), 0)
            for m in models for c in conditions
        },
        "exhausted_models": list(exhausted_models),
        "execution_stats": stats,
        "independence_audits": audits,
    }
    report_path = f"results/session_report_{day_label}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Machine-readable session report written to {report_path}")
    log("=" * 70)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(max_new_per_cell: int = 26, day_label: str = "day1") -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/raw_transcripts", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)

    log(f"Pipeline start -- max_new_per_cell={max_new_per_cell}, day={day_label}")

    client = RateLimitedGroqClient(min_request_interval=4.0, max_call_budget=100_000)
    agent = AgentLoop(client)

    all_tasks = load_tasks()
    n_trials = 2
    conditions = ["clean", "poisoned_explicit", "poisoned_implicit"]
    models_sequence = [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]

    json_path = os.path.join("results", "single_turn_results_live.json")
    csv_path = os.path.join("results", "single_turn_results_live.csv")

    # -----------------------------------------------------------------------
    # Load existing runs
    # -----------------------------------------------------------------------
    raw_evaluations: List[Dict[str, Any]] = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw_evaluations = json.load(f)
            log(f"Loaded {len(raw_evaluations)} existing runs from {json_path}.")
        except Exception as e:
            log(f"Error loading {json_path}: {e}. Starting fresh.")
            raw_evaluations = []

    completed_keys: Set[Tuple] = {
        (r.get("model", ""), r.get("condition", ""), r.get("task_id", ""), r.get("trial", 0))
        for r in raw_evaluations
    }

    # Count pre-existing runs per cell (these already count toward the cap)
    pre_existing_counts: Dict[Tuple, int] = Counter(
        (r.get("model", ""), r.get("condition", "")) for r in raw_evaluations
    )
    log(f"Pre-existing cell counts: {dict(pre_existing_counts)}")

    # New-this-session counter per (model, condition)
    new_this_session: Dict[Tuple, int] = defaultdict(int)

    # -----------------------------------------------------------------------
    # TARGET-AWARE PRE-RUN MANIFEST
    # Compute runs_needed = max(0, target - existing) per cell BEFORE any
    # API calls.  Cells with runs_needed == 0 are excluded from the
    # round-robin entirely.  Print this manifest so it can be checked by
    # hand before spending any tokens.
    # -----------------------------------------------------------------------
    target = max_new_per_cell
    log("=" * 70)
    log(f"PRE-RUN MANIFEST  (target={target} per cell)")
    log("=" * 70)
    runs_needed: Dict[Tuple, int] = {}
    total_needed = 0
    for m in models_sequence:
        for c in conditions:
            cell = (m, c)
            existing = pre_existing_counts.get(cell, 0)
            needed = max(0, target - existing)
            runs_needed[cell] = needed
            total_needed += needed
            status = "[SKIP -- already at/above target]" if needed == 0 else "need {} more (-> target {})".format(needed, target)
            log("  {:30s} | {:20s} | existing={:>3} | {}".format(m, c, existing, status))
    log(f"")
    model_token_rates = {
        "qwen/qwen3.8-27b": 6679,
        "openai/gpt-oss-20b": 2757,
        "openai/gpt-oss-120b": 3685,
    }
    model_req_rates = {
        "qwen/qwen3.8-27b": 5.73,
        "openai/gpt-oss-20b": 4.12,
        "openai/gpt-oss-120b": 5.25,
    }
    log("")
    log(f"Total new runs needed this session: {total_needed}")
    total_est_tokens = 0
    total_est_reqs = 0
    for m in models_sequence:
        m_runs = sum(runs_needed.get((m, c), 0) for c in conditions)
        m_tok = int(m_runs * model_token_rates.get(m, 3500))
        m_req = int(round(m_runs * model_req_rates.get(m, 5.0)))
        total_est_tokens += m_tok
        total_est_reqs += m_req
        log(f"  {m:30s}: {m_runs:>3} runs | ~{m_req:>3} API reqs | ~{m_tok:>7,} tokens")
    log(f"Total estimated budget: ~{total_est_reqs} API requests | ~{total_est_tokens:,} tokens")
    log("=" * 70)

    exhausted_models: Set[str] = set()
    new_runs_total = 0

    # -----------------------------------------------------------------------
    # Interleaved execution loop
    # -----------------------------------------------------------------------
    # Strategy: each "pass" selects ONE pending run per (model, condition)
    # pair, in round-robin fashion, until all cells are at cap or exhausted.

    snapshot_interval = 20  # log refill snapshot every N new runs

    while True:
        # Collect one candidate per (model, condition) that still needs runs
        round_work: List[Tuple[str, str, Dict, int]] = []
        for model in models_sequence:
            if model in exhausted_models:
                continue
            for cond in conditions:
                cell = (model, cond)
                disk_count = pre_existing_counts.get(cell, 0) + new_this_session.get(cell, 0)
                if disk_count >= max_new_per_cell:
                    continue
                # Pick the first uncompleted (task, trial) for this cell
                for t in all_tasks:
                    for trial in range(n_trials):
                        key = (model, cond, t["id"], trial)
                        if key not in completed_keys:
                            round_work.append((model, cond, t, trial))
                            break  # one per cell per pass
                    else:
                        continue
                    break

        if not round_work:
            log("All cells at target or no more work available. Exiting interleaved loop.")
            break

        # Execute this round
        for model, cond, t, trial in round_work:
            if model in exhausted_models:
                continue

            cell = (model, cond)
            disk_count = pre_existing_counts.get(cell, 0) + new_this_session.get(cell, 0)
            if disk_count >= max_new_per_cell:
                continue

            key = (model, cond, t["id"], trial)
            if key in completed_keys:
                continue

            try:
                run_res = agent.run_single_turn(
                    task=t, model=model, condition=cond, trial=trial
                )

                if run_res.get("status") == "error":
                    err_msg = run_res.get("error_message", "")
                    log(f"Run error — {model} | {cond} | {t['id']} trial {trial}: {err_msg}")
                    if any(k in err_msg.lower() for k in
                           ["daily quota", "quota exhausted", "tpd", "tokens per day", "requests per day"]):
                        log(f"Daily quota exhausted for {model}. Marking model stopped.")
                        exhausted_models.add(model)
                        break
                    # Transient error: skip this key and continue
                    completed_keys.add(key)  # prevent infinite retry
                    continue

                eval_res = evaluate_run(run_res, t)
                eval_res["trial"] = trial
                raw_evaluations.append(eval_res)
                completed_keys.add(key)
                new_this_session[cell] += 1
                new_runs_total += 1

                # Checkpoint immediately after each run
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(raw_evaluations, f, indent=2)

                # Rebuild CSV
                csv_rows = [
                    {
                        "model": r.get("model", ""),
                        "condition": r.get("condition", ""),
                        "task_id": r.get("task_id", ""),
                        "trial": r.get("trial", 0),
                        "task_completed": int(r.get("task_completed", 0)),
                        "attack_succeeded": int(r.get("attack_succeeded", 0)),
                        "num_tool_calls": r.get("num_tool_calls", 0),
                        "avg_brier_score": r.get("avg_brier_score", 0.0),
                    }
                    for r in raw_evaluations
                ]
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(csv_rows)

                # Save client stats snapshot
                stats = client.get_execution_stats()
                with open("results/execution_stats.json", "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)

                current_cell_total = pre_existing_counts.get(cell, 0) + new_this_session.get(cell, 0)
                log(
                    f"[+{new_runs_total} sess | {len(raw_evaluations)} disk] "
                    f"{model} | {cond} | {t['id']} tr{trial} → "
                    f"cell={current_cell_total}/{max_new_per_cell} | "
                    f"Comp={eval_res['task_completed']} ASR={eval_res['attack_succeeded']} "
                    f"Tools={eval_res['num_tool_calls']} Brier={eval_res['avg_brier_score']:.4f}"
                )

                # Periodic token-refill snapshot
                if new_runs_total % snapshot_interval == 0:
                    log_refill_snapshot(client, label=f"after {new_runs_total} new runs")

            except Exception as e:
                err_str = str(e)
                log(f"EXCEPTION — {model} | {cond} | {t['id']} trial {trial}: {err_str[:200]}")
                if any(k in err_str.lower() for k in
                       ["daily quota", "quota exhausted", "tpd", "tokens per day", "requests per day"]):
                    log(f"Daily quota exhausted for {model}. Marking model stopped.")
                    exhausted_models.add(model)
                    break
                else:
                    # Mark key done to avoid hammering the same failing run
                    completed_keys.add(key)
                    log("Skipping this run and continuing.")

    # -----------------------------------------------------------------------
    # End-of-session report (no multi-turn in this script;
    # multi-turn runs in a separate script once single-turn is balanced)
    # -----------------------------------------------------------------------
    write_end_of_session_report(
        records=raw_evaluations,
        client=client,
        new_runs_this_session=new_runs_total,
        planned_new_per_cell=max_new_per_cell,
        day_label=day_label,
        exhausted_models=exhausted_models,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interleaved balanced single-turn sweep"
    )
    parser.add_argument(
        "--max_new_per_cell",
        type=int,
        default=26,
        help="Stop adding runs once a cell reaches this count on disk. Default: 26.",
    )
    parser.add_argument(
        "--day",
        type=str,
        default="day1",
        help="Day label for session report filename. E.g. day1, day2.",
    )
    args = parser.parse_args()
    run_pipeline(max_new_per_cell=args.max_new_per_cell, day_label=args.day)
