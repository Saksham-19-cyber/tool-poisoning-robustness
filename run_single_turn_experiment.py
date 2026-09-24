import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List
from harness.agent_loop import AgentLoop
from harness.groq_client import RateLimitedGroqClient
from harness.scorer import evaluate_run, aggregate_experiment_results
from tasks.task_loader import load_tasks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--condition", type=str, default="all")
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    client = RateLimitedGroqClient()
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
        all_tasks = all_tasks[:args.limit_tasks]

    agent = AgentLoop(client)
    raw_evaluations: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    print(f"Starting single-turn run with {len(models_to_run)} models, {len(conditions)} conditions, {len(all_tasks)} tasks.")

    for model_name in models_to_run:
        for cond in conditions:
            print(f"Running model={model_name} condition={cond} ...")
            cond_records = []
            for t in all_tasks:
                run_res = agent.run_single_turn(
                    task=t,
                    model=model_name,
                    condition=cond
                )
                eval_res = evaluate_run(run_res, t)
                raw_evaluations.append(eval_res)
                cond_records.append(eval_res)

                csv_rows.append({
                    "model": model_name,
                    "condition": cond,
                    "task_id": t["id"],
                    "task_completed": eval_res["task_completed"],
                    "attack_succeeded": eval_res["attack_succeeded"],
                    "num_tool_calls": eval_res["num_tool_calls"],
                    "avg_brier_score": round(eval_res["avg_brier_score"], 4)
                })

            summary = aggregate_experiment_results(cond_records)
            print(f"Summary [{model_name} | {cond}]: ASR={summary.get('attack_success_rate')} Utility={summary.get('task_utility')} Brier={summary.get('mean_brier_score')}")

    json_path = os.path.join(args.output_dir, "single_turn_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_evaluations, f, indent=2)

    csv_path = os.path.join(args.output_dir, "single_turn_results.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"Completed single-turn experiments. Total Groq API calls: {client.total_calls}")
    print(f"Results saved to {json_path} and {csv_path}")

if __name__ == "__main__":
    main()
