import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List
from harness.groq_client import RateLimitedGroqClient
from harness.multi_turn_runner import MultiTurnRunner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--condition", type=str, default="poisoned_explicit")
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

    scenarios_path = os.path.join(os.path.dirname(__file__), "tasks", "multi_turn_scenarios.json")
    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    runner = MultiTurnRunner(client)
    all_results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    print(f"Starting multi-turn runs with {len(models_to_run)} models, condition={args.condition}, {len(scenarios)} scenarios.")

    for model_name in models_to_run:
        for scen in scenarios:
            scen_id = scen.get("scenario_id")
            print(f"Executing scenario={scen_id} model={model_name} condition={args.condition} ...")
            res = runner.run_scenario(
                scenario=scen,
                model=model_name,
                condition=args.condition
            )
            all_results.append(res)

            for t in res.get("turns", []):
                csv_rows.append({
                    "model": model_name,
                    "condition": args.condition,
                    "scenario_id": scen_id,
                    "turn": t["turn"],
                    "avg_confidence": t["avg_confidence"],
                    "confidence_prob": t["confidence_prob"],
                    "turn_correct": t["turn_correct"],
                    "attack_succeeded": t["attack_succeeded"],
                    "task_completed": t["task_completed"],
                    "brier_score": t["brier_score"],
                    "drift_slope": res["calibration_drift_slope"]
                })

            print(f"Finished {scen_id}: drift_slope={res['calibration_drift_slope']} mean_brier={res['mean_brier']}")

    json_path = os.path.join(args.output_dir, "multi_turn_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    csv_path = os.path.join(args.output_dir, "multi_turn_results.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"Completed multi-turn experiments. Total Groq API calls: {client.total_calls}")
    print(f"Results saved to {json_path} and {csv_path}")

if __name__ == "__main__":
    main()
