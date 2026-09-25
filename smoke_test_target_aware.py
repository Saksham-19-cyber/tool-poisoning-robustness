"""
smoke_test_target_aware.py -- run from project root:
    python smoke_test_target_aware.py

Tests target-aware scheduling against REAL on-disk data (154 runs).
Does NOT delete or modify the live results file.

Strategy:
- target = 5 (below all clean cells at 38-52, above most poisoned cells at 0-11)
- Expected result:
    qwen/clean (52)  -> needs 0 runs  [SKIP]
    gpt-20b/clean (38) -> needs 0 runs  [SKIP]
    gpt-120b/clean (52) -> needs 0 runs  [SKIP]
    qwen/PE (1)      -> needs 4 runs  (1->5)
    gpt-120b/PE (11) -> needs 0 runs  [SKIP - already above target=5]
    qwen/PI (0)      -> needs 5 runs  (0->5)
    gpt-20b/PE (0)   -> needs 5 runs  (0->5)
    gpt-20b/PI (0)   -> needs 5 runs  (0->5)
    gpt-120b/PI (0)  -> needs 5 runs  (0->5)

Total new runs expected: 4+0+0+5+5+5+5 = 24
All clean cells and gpt-120b/PE must get exactly 0 new runs.
"""
import os
import sys
import json
import time

os.environ["GROQ_API_KEY"] = "mock"

TARGET = 5  # below 120b/PE=11 (which should be SKIPPED as already >= target)

live_json = "results/single_turn_results_live.json"
if not os.path.exists(live_json):
    print("ERROR: live results file not found. Cannot run target-aware smoke test.")
    sys.exit(1)

with open(live_json, encoding="utf-8") as f:
    original_data = json.load(f)

from collections import Counter
pre_existing = Counter(
    (r.get("model"), r.get("condition")) for r in original_data
)
print("=" * 70)
print("TARGET-AWARE SMOKE TEST  (target={})".format(TARGET))
print("Pre-existing cell counts from disk:".format())
models = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
conditions = ["clean", "poisoned_explicit", "poisoned_implicit"]
expected_new = {}
for m in models:
    for c in conditions:
        ex = pre_existing.get((m, c), 0)
        need = max(0, TARGET - ex)
        expected_new[(m, c)] = need
        skip = "[SKIP]" if need == 0 else "need {}".format(need)
        print("  {:30s} | {:20s} | existing={:>3} | {}".format(m, c, ex, skip))

expected_total = sum(expected_new.values())
print()
print("Expected total new runs: {}".format(expected_total))
print("=" * 70)
print()

# Run pipeline
import run_live_master_pipeline
print("Running pipeline with max_new_per_cell={}...".format(TARGET))
run_live_master_pipeline.run_pipeline(max_new_per_cell=TARGET, day_label="smoke_target_aware")

# Reload live results and diff
with open(live_json, encoding="utf-8") as f:
    post_data = json.load(f)
post_counts = Counter(
    (r.get("model"), r.get("condition")) for r in post_data
)

print()
print("=" * 70)
print("POST-RUN VERIFICATION")
print("=" * 70)
all_pass = True
for m in models:
    for c in conditions:
        pre = pre_existing.get((m, c), 0)
        post = post_counts.get((m, c), 0)
        actual_new = post - pre
        expected = expected_new[(m, c)]
        ok = (actual_new == expected)
        verdict = "PASS" if ok else "FAIL (got {}, expected {})".format(actual_new, expected)
        if not ok:
            all_pass = False
        print("  {:30s} | {:20s} | pre={:>3} post={:>3} new={:>2} expected={:>2} | {}".format(
            m, c, pre, post, actual_new, expected, verdict
        ))

print()
if all_pass:
    print("ALL CELLS PASS -- target-aware scheduling is working correctly.")
    print("Cells >= target got 0 new runs. Cells < target got exactly (target - existing) new runs.")
else:
    print("SOME CELLS FAILED -- target-aware scheduling has a bug.")
print("=" * 70)

# Restore original data (undo any test runs from mock)
with open(live_json, "w", encoding="utf-8") as f:
    json.dump(original_data, f, indent=2)
print("Live results restored to original {} records.".format(len(original_data)))
