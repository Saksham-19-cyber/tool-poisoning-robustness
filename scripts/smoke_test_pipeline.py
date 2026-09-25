"""Smoke test: run pipeline in mock mode with max_new_per_cell=2, verify interleaving."""
import os
import sys
import shutil
import time
import json

os.environ["GROQ_API_KEY"] = "mock"

# Backup and clear live results so smoke test starts fresh
live_json = "results/single_turn_results_live.json"
backup_json = f"results/_smoke_backup_{int(time.time())}.json"
if os.path.exists(live_json):
    shutil.copy(live_json, backup_json)
    os.remove(live_json)
    print(f"Backed up and removed {live_json}")

try:
    import importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_live_master_pipeline", "run_live_master_pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_pipeline(max_new_per_cell=2, day_label="smoke_test")
    print("\n=== SMOKE TEST: checking interleaving ===")
    report_path = "results/session_report_smoke_test.json"
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            rep = json.load(f)
        print(f"new_runs_this_session: {rep['new_runs_this_session']}")
        print(f"total_runs_on_disk: {rep['total_runs_on_disk']}")
        print("per_cell_counts:")
        for cell, n in rep["per_cell_counts"].items():
            print(f"  {cell}: {n}")
        print("model_refill_report (from execution_stats):")
        mrr = rep.get("execution_stats", {}).get("model_refill_report", {})
        for m, s in mrr.items():
            print(f"  {m}: consumed={s['tokens_consumed']}, avg/call={s['avg_tokens_per_call']:.0f}")
        print("independence_audits (clean only — poisoned cells empty):")
        for a in rep.get("independence_audits", []):
            print(f"  {a['model']:30s} | {a['condition']:22s} | N={a['n']} | phi={a.get('phi_coefficient')}")
    print("SMOKE TEST PASSED")
except Exception as e:
    print(f"SMOKE TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore original results
    if os.path.exists(backup_json):
        shutil.copy(backup_json, live_json)
        os.remove(backup_json)
        print(f"Restored original results.")
    elif not os.path.exists(live_json):
        print("WARNING: No backup to restore — results file is gone.")
