"""
scripts/independence_audit.py
==============================
Re-run the ASR / task-completion independence audit (Item 1 methodology)
on the current results file.

Usage:
    python scripts/independence_audit.py [--results path/to/results.json] [--min_n 5]

Reports per (model, condition) cell:
  - N rows
  - per-row table: attack_succeeded vs task_completed
  - both_true, both_false, attack_only (ASR=T, task=F), task_only (ASR=F, task=T)
  - phi (Matthews correlation coefficient)
  - Verdict: DEGENERATE | STRONGLY_CORRELATED | MODERATELY_CORRELATED | NEAR_INDEPENDENT

This output belongs in the Limitations/Methodology section regardless of outcome.
"""

import argparse
import json
import math
import sys
from collections import Counter

# Force UTF-8 output on Windows (prevents UnicodeEncodeError in print())
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
CONDITIONS = ["clean", "poisoned_explicit", "poisoned_implicit"]


def phi_coefficient(tp, fp, fn, tn):
    """Matthews correlation coefficient for 2x2 confusion table."""
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return None
    return (tp * tn - fp * fn) / math.sqrt(denom)


def audit_cell(rows, model, condition):
    subset = [
        r for r in rows
        if r.get("model") == model and r.get("condition") == condition
    ]
    n = len(subset)
    if n == 0:
        return None

    both_true = sum(1 for r in subset if r.get("attack_succeeded") and r.get("task_completed"))
    both_false = sum(1 for r in subset if not r.get("attack_succeeded") and not r.get("task_completed"))
    attack_only = sum(1 for r in subset if r.get("attack_succeeded") and not r.get("task_completed"))
    task_only = sum(1 for r in subset if not r.get("attack_succeeded") and r.get("task_completed"))

    phi = phi_coefficient(both_true, attack_only, task_only, both_false)

    if attack_only == 0 and task_only == 0:
        verdict = "DEGENERATE (perfect co-occurrence, no discriminating rows)"
    elif phi is None:
        verdict = "UNDEFINED (trivial table)"
    elif abs(phi) >= 0.9:
        verdict = "STRONGLY_CORRELATED (phi={:.3f})".format(phi)
    elif abs(phi) >= 0.5:
        verdict = "MODERATELY_CORRELATED (phi={:.3f})".format(phi)
    else:
        verdict = "WEAKLY_CORRELATED / NEAR_INDEPENDENT (phi={:.3f})".format(phi)

    return {
        "model": model,
        "condition": condition,
        "n": n,
        "both_true": both_true,
        "both_false": both_false,
        "attack_only": attack_only,
        "task_only": task_only,
        "phi": round(phi, 4) if phi is not None else None,
        "verdict": verdict,
        "rows": [
            {
                "task_id": r.get("task_id"),
                "trial": r.get("trial"),
                "attack_succeeded": r.get("attack_succeeded"),
                "task_completed": r.get("task_completed"),
                "num_tool_calls": r.get("num_tool_calls", 0),
            }
            for r in subset
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="ASR/task-completion independence audit")
    parser.add_argument(
        "--results",
        default="results/single_turn_results_live.json",
        help="Path to results JSON file.",
    )
    parser.add_argument(
        "--min_n",
        type=int,
        default=1,
        help="Skip cells with fewer than this many rows.",
    )
    parser.add_argument(
        "--per_row",
        action="store_true",
        default=False,
        help="Print full per-row table for each cell.",
    )
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        rows = json.load(f)

    print("Loaded {} records from {}".format(len(rows), args.results))
    print("Cell counts: {}".format(dict(Counter((r.get("model"), r.get("condition")) for r in rows))))
    print()
    print("=" * 90)
    print("ASR / TASK-COMPLETION INDEPENDENCE AUDIT")
    print("Item 1 methodology -- must appear in Limitations/Methodology section")
    print("=" * 90)
    print()

    audits = []
    for model in MODELS:
        for condition in CONDITIONS:
            result = audit_cell(rows, model, condition)
            if result is None or result["n"] < args.min_n:
                continue
            audits.append(result)

            print("-" * 90)
            print("Model:     {}".format(model))
            print("Condition: {}  |  N={}".format(condition, result["n"]))
            print()

            if args.per_row:
                hdr = "  {:>3}  {:<15}  {:>5}  {:>16}  {:>14}  {:>10}".format(
                    "#", "task_id", "trial", "attack_succeeded", "task_completed", "tool_calls"
                )
                print(hdr)
                print("  " + "-" * (len(hdr) - 2))
                for i, r in enumerate(result["rows"], 1):
                    print(
                        "  {:>3}  {:<15}  {:>5}  {:>16}  {:>14}  {:>10}".format(
                            i,
                            str(r["task_id"]),
                            str(r["trial"]),
                            str(r["attack_succeeded"]),
                            str(r["task_completed"]),
                            r["num_tool_calls"],
                        )
                    )
                print()

            print("  Confusion table:")
            print("                      task_completed=T   task_completed=F")
            print("    attack_succeeded=T   {:>5}              {:>5}  <- attack-only (discordant)".format(
                result["both_true"], result["attack_only"]
            ))
            print("    attack_succeeded=F   {:>5}              {:>5}".format(
                result["task_only"], result["both_false"]
            ))
            print("                         ^ task-only (discordant)")
            print()
            print("  both_true={}  both_false={}  attack_only={}  task_only={}".format(
                result["both_true"], result["both_false"],
                result["attack_only"], result["task_only"]
            ))
            print("  phi (Matthews CC) = {}".format(result["phi"]))
            print("  VERDICT: {}".format(result["verdict"]))
            print()

    print("=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print("{:30s}  {:22s}  {:>4}  {:>6}  {}".format("Model", "Condition", "N", "phi", "Verdict"))
    print("-" * 90)
    for a in audits:
        phi_str = "{:+.3f}".format(a["phi"]) if a["phi"] is not None else "  N/A"
        print("{:30s}  {:22s}  {:>4}  {:>6}  {}".format(
            a["model"], a["condition"], a["n"], phi_str, a["verdict"]
        ))

    print()
    degenerate_cells = [a for a in audits if "DEGENERATE" in a["verdict"]]
    if degenerate_cells:
        print("WARNING: {} cell(s) are DEGENERATE -- ASR and task_completed".format(len(degenerate_cells)))
        print("         co-vary perfectly (every row either both True or both False).")
        print("         These cells cannot separately estimate utility and robustness.")
        print("         Disclose explicitly in Limitations section.")
    else:
        print("No degenerate cells detected at this sample size.")

    # Write JSON audit report
    out_path = "results/independence_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2)
    print("\nFull audit written to {}".format(out_path))


if __name__ == "__main__":
    main()
