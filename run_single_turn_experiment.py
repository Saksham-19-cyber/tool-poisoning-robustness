import argparse
import csv
import json
import os
from typing import Any, Dict, List

from harness.agent_loop import AgentLoop
from harness.groq_client import RateLimitedGroqClient
from harness.scorer import (
    aggregate_experiment_results,
    evaluate_run,
    two_proportion_z_test,
    wilson_ci,
)
from tasks.task_loader import load_tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--condition", type=str, default="all")
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=2,
                        help="Repeated trials per task (enables statistical CI estimation). Default=2.")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    client = RateLimitedGroqClient(max_call_budget=10000)
    models_dict = client.select_model_matrix()

    if args.model == "all":
        models_to_run = [models_dict["small"], models_dict["mid"], models_dict["large"]]
    elif args.model in models_dict:
        models_to_run = [models_dict[args.model]]
    else:
        models_to_run = [args.model]

    if args.condition == "all":
        conditions = ["clean", "poisoned_explicit", "poisoned_implicit"]
    else:
        conditions = [args.condition]

    all_tasks = load_tasks()
    if args.limit_tasks is not None:
        all_tasks = all_tasks[: args.limit_tasks]

    n_trials = max(1, args.n_trials)
    total_runs_per_cell = len(all_tasks) * n_trials
    agent = AgentLoop(client)

    total_planned = len(models_to_run) * len(conditions) * total_runs_per_cell
    print(
        f"Starting single-turn experiment: {len(models_to_run)} models × "
        f"{len(conditions)} conditions × {len(all_tasks)} tasks × {n_trials} trial(s) "
        f"= {total_planned} total runs.",
        flush=True
    )

    json_path = os.path.join(args.output_dir, "single_turn_results_live.json")
    csv_path = os.path.join(args.output_dir, "single_turn_results_live.csv")

    raw_evaluations: List[Dict[str, Any]] = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw_evaluations = json.load(f)
            print(f"Loaded {len(raw_evaluations)} existing runs from {json_path}. Resuming...", flush=True)
        except Exception:
            raw_evaluations = []

    completed_keys = {
        (r.get("model", ""), r.get("condition", ""), r.get("task_id", ""), r.get("trial", 0))
        for r in raw_evaluations
    }

    csv_rows: List[Dict[str, Any]] = []
    for r in raw_evaluations:
        csv_rows.append(
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
        )

    summary_rows: List[Dict[str, Any]] = []
    run_counter = len(completed_keys)

    for model_name in models_to_run:
        for cond in conditions:
            print(f"\n==================================================", flush=True)
            print(f"  Starting Cell: model={model_name} | condition={cond}", flush=True)
            print(f"==================================================", flush=True)
            cond_records: List[Dict[str, Any]] = []

            for t in all_tasks:
                for trial in range(n_trials):
                    key = (model_name, cond, t["id"], trial)
                    if key in completed_keys:
                        existing = next(
                            r for r in raw_evaluations
                            if (r.get("model") == model_name and r.get("condition") == cond
                                and r.get("task_id") == t["id"] and r.get("trial", 0) == trial)
                        )
                        cond_records.append(existing)
                        continue

                    run_counter += 1
                    run_res = agent.run_single_turn(task=t, model=model_name, condition=cond, trial=trial)
                    eval_res = evaluate_run(run_res, t)
                    eval_res["trial"] = trial
                    raw_evaluations.append(eval_res)
                    cond_records.append(eval_res)
                    completed_keys.add(key)

                    csv_rows.append(
                        {
                            "model": model_name,
                            "condition": cond,
                            "task_id": t["id"],
                            "trial": trial,
                            "task_completed": int(eval_res["task_completed"]),
                            "attack_succeeded": int(eval_res["attack_succeeded"]),
                            "num_tool_calls": eval_res["num_tool_calls"],
                            "avg_brier_score": eval_res["avg_brier_score"],
                        }
                    )

                    print(
                        f"  [{run_counter:3d}/{total_planned}] {t['id']} (trial {trial}): "
                        f"Completed={eval_res['task_completed']} | Hijacked={eval_res['attack_succeeded']} | "
                        f"Tools={eval_res['num_tool_calls']} | Brier={eval_res['avg_brier_score']:.4f}",
                        flush=True
                    )

                    # Save checkpoint after each trial
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(raw_evaluations, f, indent=2)
                    if csv_rows:
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                            writer.writeheader()
                            writer.writerows(csv_rows)

            summary = aggregate_experiment_results(cond_records)
            n = summary["total_runs"]
            asr = summary["attack_success_rate"]
            asr_lo = summary["asr_ci_lo"]
            asr_hi = summary["asr_ci_hi"]
            util = summary["task_utility"]
            util_lo = summary["utility_ci_lo"]
            util_hi = summary["utility_ci_hi"]
            brier = summary["mean_brier_score"]
            brier_lo = summary["brier_ci_lo"]
            brier_hi = summary["brier_ci_hi"]
            ece = summary["expected_calibration_error"]
            ece_lo = summary.get("ece_ci_lo", 0.0)
            ece_hi = summary.get("ece_ci_hi", 0.0)

            print(
                f"\n  [Cell Summary] {model_name} ({cond}):\n"
                f"    ASR  = {asr:.3f}  95% CI [{asr_lo:.3f}, {asr_hi:.3f}]  (n={n})\n"
                f"    Util = {util:.3f}  95% CI [{util_lo:.3f}, {util_hi:.3f}]\n"
                f"    Brier= {brier:.4f}  95% CI [{brier_lo:.4f}, {brier_hi:.4f}]\n"
                f"    ECE  = {ece:.4f}  95% CI [{ece_lo:.4f}, {ece_hi:.4f}]",
                flush=True
            )

            summary_rows.append(
                {
                    "model": model_name,
                    "condition": cond,
                    "total_runs": n,
                    "n_attack": summary["n_attack_succeeded"],
                    "n_util": summary["n_task_completed"],
                    "asr": asr,
                    "asr_ci_lo": asr_lo,
                    "asr_ci_hi": asr_hi,
                    "task_utility": util,
                    "utility_ci_lo": util_lo,
                    "utility_ci_hi": util_hi,
                    "mean_brier": brier,
                    "brier_ci_lo": brier_lo,
                    "brier_ci_hi": brier_hi,
                    "ece": ece,
                    "ece_ci_lo": ece_lo,
                    "ece_ci_hi": ece_hi,
                }
            )

    _print_significance_tests(summary_rows)

    json_path = os.path.join(args.output_dir, "single_turn_results_live.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_evaluations, f, indent=2)

    csv_path = os.path.join(args.output_dir, "single_turn_results_live.csv")
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    summary_path = os.path.join("analysis", "summary_table.csv")
    os.makedirs("analysis", exist_ok=True)
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    stats_path = os.path.join(args.output_dir, "execution_stats.json")
    try:
        stats = client.get_execution_stats()
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Warning: could not write execution stats: {e}")

    print(
        f"\nDone. Total Groq API calls: {client.total_calls}\n"
        f"Results: {json_path}, {csv_path}\n"
        f"Summary: {summary_path}\n"
        f"Stats: {stats_path}"
    )


def _print_significance_tests(summary_rows: List[Dict[str, Any]]) -> None:
    print("\n--- Pairwise ASR significance tests (two-proportion z-test) ---")
    for model_group in _group_by_model(summary_rows):
        model = model_group[0]["model"]
        explicit = next((r for r in model_group if r["condition"] == "poisoned_explicit"), None)
        implicit = next((r for r in model_group if r["condition"] == "poisoned_implicit"), None)
        if explicit and implicit:
            p = two_proportion_z_test(
                explicit["n_attack"], explicit["total_runs"],
                implicit["n_attack"], implicit["total_runs"],
            )
            sig = "[p<0.05]" if p < 0.05 else "[n.s.]"
            print(
                f"  {model}: explicit_ASR={explicit['asr']:.3f} vs implicit_ASR={implicit['asr']:.3f}"
                f"  p={p:.3f}  {sig}"
            )

    print("\n--- Cross-model explicit ASR tests ---")
    explicit_rows = [r for r in summary_rows if r["condition"] == "poisoned_explicit"]
    for i in range(len(explicit_rows)):
        for j in range(i + 1, len(explicit_rows)):
            r1, r2 = explicit_rows[i], explicit_rows[j]
            p = two_proportion_z_test(r1["n_attack"], r1["total_runs"], r2["n_attack"], r2["total_runs"])
            sig = "[p<0.05]" if p < 0.05 else "[n.s.]"
            print(
                f"  {r1['model']} ({r1['asr']:.3f}) vs {r2['model']} ({r2['asr']:.3f})"
                f"  p={p:.3f}  {sig}"
            )


def _group_by_model(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    seen = {}
    for r in rows:
        seen.setdefault(r["model"], []).append(r)
    return list(seen.values())


if __name__ == "__main__":
    main()
