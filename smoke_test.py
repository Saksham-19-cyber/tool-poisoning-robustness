"""
smoke_test.py — run from project root:
    python smoke_test.py
"""
import os
import sys
import shutil
import time
import json

os.environ["GROQ_API_KEY"] = "mock"

live_json = "results/single_turn_results_live.json"
backup_json = "results/_smoke_backup.json"

if os.path.exists(live_json):
    shutil.copy(live_json, backup_json)
    os.remove(live_json)
    print("Backed up and removed live results.")

exit_code = 0
try:
    import run_live_master_pipeline
    run_live_master_pipeline.run_pipeline(max_new_per_cell=2, day_label="smoke_test")

    print()
    print("=== INTERLEAVING CHECK ===")
    report_path = "results/session_report_smoke_test.json"
    with open(report_path, encoding="utf-8") as f:
        rep = json.load(f)

    print("new_runs_this_session:", rep["new_runs_this_session"])
    print("total_runs_on_disk:", rep["total_runs_on_disk"])
    print("per_cell_counts:")
    for cell, n in rep["per_cell_counts"].items():
        print(f"  {cell}: {n}")

    mrr = rep.get("execution_stats", {}).get("model_refill_report", {})
    print("model_refill_report:")
    for m, s in mrr.items():
        tok = s["tokens_consumed"]
        apc = s["avg_tokens_per_call"]
        hits = s["tpd_hits"]
        print(f"  {m}: consumed={tok}, avg/call={apc:.0f}, tpd_hits={hits}")

    print("independence_audits:")
    for a in rep.get("independence_audits", []):
        mod = a["model"][:25]
        cond = a["condition"]
        n = a.get("n_rows") or a.get("n", "?")
        phi = a.get("phi_coefficient") or a.get("phi")
        print("  {:25s} | {:22s} | N={} | phi={}".format(mod, cond, n, phi))

    # Verify interleaving: no single condition should dominate
    # All cells should have count <= 2, and multiple cells should be populated
    counts_by_cond = {}
    for cell_key, n in rep["per_cell_counts"].items():
        parts = cell_key.split("|")
        if len(parts) == 2:
            cond = parts[1]
            counts_by_cond[cond] = counts_by_cond.get(cond, 0) + n

    print()
    print("Condition totals (should be balanced):", counts_by_cond)
    max_c = max(counts_by_cond.values()) if counts_by_cond else 0
    min_c = min(counts_by_cond.values()) if counts_by_cond else 0
    imbalance = max_c - min_c
    if imbalance <= 2:
        print(f"INTERLEAVING OK — max imbalance across conditions: {imbalance}")
    else:
        print(f"WARNING — condition imbalance = {imbalance} (expected <= 2 for mock run)")

    print()
    print("SMOKE TEST PASSED")

except Exception as e:
    import traceback
    print("SMOKE TEST FAILED:", e)
    traceback.print_exc()
    exit_code = 1

finally:
    if os.path.exists(backup_json):
        shutil.copy(backup_json, live_json)
        os.remove(backup_json)
        print("Restored original live results.")

sys.exit(exit_code)
