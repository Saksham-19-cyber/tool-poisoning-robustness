"""
Dump full record for each openai/gpt-oss-120b | poisoned_explicit row,
including all step_evals to show independent signal computation.
"""
import json

with open("results/single_turn_results_live.json") as f:
    d = json.load(f)

rows = [r for r in d
        if r.get("model") == "openai/gpt-oss-120b"
        and r.get("condition") == "poisoned_explicit"]

for i, r in enumerate(rows, 1):
    tid = r.get("task_id", "?")
    trial = r.get("trial", "?")
    print(f"--- ROW {i}: task={tid}  trial={trial} ---")
    se = r.get("step_evals")
    if se:
        print(f"  step_evals ({len(se)} steps):")
        for j, step in enumerate(se):
            print(f"    step {j}: {json.dumps(step)}")
    for k, v in r.items():
        if k != "step_evals":
            print(f"  {k}: {repr(v)}")
    print()
