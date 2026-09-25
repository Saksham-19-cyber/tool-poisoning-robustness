"""
Print per-run detail for openai/gpt-oss-120b | poisoned_explicit
and show what signals feed attack_succeeded vs task_completed.
"""
import json

with open("results/single_turn_results_live.json") as f:
    d = json.load(f)

rows = [r for r in d
        if r.get("model") == "openai/gpt-oss-120b"
        and r.get("condition") == "poisoned_explicit"]

print(f"=== openai/gpt-oss-120b | poisoned_explicit  ({len(rows)} rows) ===\n")
header = (
    f"{'#':>3}  {'task_id':<28}  {'trial':>5}  "
    f"{'attack_succeeded':>16}  {'task_completed':>14}  "
    f"{'confidence':>10}  {'brier':>7}  {'steps':>5}  "
    f"{'tool_calls':>10}"
)
print(header)
print("-" * len(header))

for i, r in enumerate(rows, 1):
    print(
        f"{i:>3}  {str(r.get('task_id','?')):<28}  "
        f"{str(r.get('trial','?')):>5}  "
        f"{str(r.get('attack_succeeded','?')):>16}  "
        f"{str(r.get('task_completed','?')):>14}  "
        f"{str(r.get('confidence','?')):>10}  "
        f"{str(r.get('brier_score','?')):>7}  "
        f"{str(r.get('num_steps','?')):>5}  "
        f"{str(r.get('tool_calls_made','?')):>10}"
    )

print()

# Show ALL keys present in these rows so we can see every signal
print("=== ALL KEYS IN THESE ROWS ===")
all_keys = set()
for r in rows:
    all_keys.update(r.keys())
print(sorted(all_keys))

print()
# Print attack_decision_log if present for first few rows
for i, r in enumerate(rows, 1):
    log = r.get("attack_decision_log") or r.get("attack_log") or r.get("attack_reason")
    if log:
        print(f"Row {i} attack_decision_log: {log}")
    malicious = r.get("malicious_tool_calls") or r.get("poisoned_tool_invocations")
    if malicious:
        print(f"Row {i} malicious_tool_calls: {malicious}")
