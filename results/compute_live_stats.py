import json
import sys
sys.path.insert(0, ".")
from harness.scorer import aggregate_experiment_results, two_proportion_z_test

d = json.load(open('results/single_turn_results_live.json'))
cells = {}
for r in d:
    key = (r['model'], r['condition'])
    cells.setdefault(key, []).append(r)

summaries = {}
for (model, cond), records in cells.items():
    s = aggregate_experiment_results(records)
    summaries[(model, cond)] = s
    n = s['total_runs']
    print(f"=== Cell: {model} | {cond} (N={n}) ===")
    print(f"  Task Utility (Acc):   {s['task_utility']:.4f}  95% CI [{s['utility_ci_lo']:.4f}, {s['utility_ci_hi']:.4f}]  (n={s['n_task_completed']}/{n})")
    print(f"  Attack Success (ASR): {s['attack_success_rate']:.4f}  95% CI [{s['asr_ci_lo']:.4f}, {s['asr_ci_hi']:.4f}]  (n={s['n_attack_succeeded']}/{n})")
    print(f"  Brier Score:          {s['mean_brier_score']:.4f}  95% CI [{s['brier_ci_lo']:.4f}, {s['brier_ci_hi']:.4f}]")
    print(f"  ECE:                  {s['expected_calibration_error']:.4f}  95% CI [{s['ece_ci_lo']:.4f}, {s['ece_ci_hi']:.4f}]")
    print()

# Two-proportion z-tests
print("=== Pairwise Significance Tests (two-proportion z-test) ===")
g120_clean = summaries.get(('openai/gpt-oss-120b', 'clean'))
g120_exp = summaries.get(('openai/gpt-oss-120b', 'poisoned_explicit'))
if g120_clean and g120_exp:
    p_val = two_proportion_z_test(g120_exp['n_attack_succeeded'], g120_exp['total_runs'], g120_clean['n_attack_succeeded'], g120_clean['total_runs'])
    sig = "[p < 0.001 ***]" if p_val < 0.001 else ("[p < 0.05 *]" if p_val < 0.05 else "[n.s.]")
    print(f"gpt-oss-120b Explicit ASR ({g120_exp['attack_success_rate']:.3f}, n={g120_exp['total_runs']}) vs Clean ASR ({g120_clean['attack_success_rate']:.3f}, n={g120_clean['total_runs']}):")
    print(f"  z-test p-value: {p_val:.4e} {sig}")

# Save detailed summary json
with open('results/statistical_summary_live.json', 'w') as f:
    json.dump({f"{k[0]}__{k[1]}": v for k, v in summaries.items()}, f, indent=2)
print("\nSaved statistical summary to results/statistical_summary_live.json")
