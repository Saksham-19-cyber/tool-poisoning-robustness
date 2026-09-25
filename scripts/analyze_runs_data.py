import json
import glob
import os
from collections import defaultdict

def main():
    print("=== ANALYSIS OF REAL RUN DATA ===")
    
    # 1. execution_stats.json
    stats_file = "results/execution_stats.json"
    if os.path.exists(stats_file):
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        print("\n--- execution_stats.json ---")
        for k in ["wall_clock_seconds", "total_calls", "total_prompt_tokens", "total_completion_tokens", "total_tokens"]:
            print(f"  {k}: {stats.get(k)}")
        refill = stats.get("model_refill_report", {})
        print("  model_refill_report:")
        print(json.dumps(refill, indent=4))
    else:
        print("\n--- No execution_stats.json found ---")

    # 2. single_turn_results_live.json
    res_file = "results/single_turn_results_live.json"
    if os.path.exists(res_file):
        with open(res_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"\n--- single_turn_results_live.json (total {len(records)} records) ---")
        
        by_cell = defaultdict(list)
        for r in records:
            key = (r.get("model"), r.get("condition"))
            by_cell[key].append(r)
        
        all_tool_calls = []
        for (model, cond), cell_records in sorted(by_cell.items()):
            tcs = [r.get("num_tool_calls", 0) for r in cell_records]
            all_tool_calls.extend(tcs)
            mean_tc = sum(tcs) / len(tcs) if tcs else 0
            print(f"  {model} | {cond}: N={len(cell_records)}, tool_calls: mean={mean_tc:.2f}, min={min(tcs)}, max={max(tcs)}")
        
        mean_all_tc = sum(all_tool_calls) / len(all_tool_calls) if all_tool_calls else 0
        print(f"\n  OVERALL across {len(records)} runs:")
        print(f"  mean tool_calls per run: {mean_all_tc:.3f}")
    
    # 3. raw transcripts analysis
    raw_files = glob.glob("results/raw_transcripts/*.json")
    print(f"\n--- raw_transcripts analysis (total {len(raw_files)} files) ---")
    by_model_usage = defaultdict(lambda: {"prompt": [], "completion": [], "total": [], "calls": 0})
    
    for rf in raw_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                d = json.load(f)
            model = d.get("model", "unknown")
            usage = d.get("usage", {})
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            tt = usage.get("total_tokens", 0)
            if tt > 0:
                by_model_usage[model]["prompt"].append(pt)
                by_model_usage[model]["completion"].append(ct)
                by_model_usage[model]["total"].append(tt)
                by_model_usage[model]["calls"] += 1
        except Exception:
            pass

    for model, u in sorted(by_model_usage.items()):
        n = u["calls"]
        if n > 0:
            avg_pt = sum(u["prompt"]) / n
            avg_ct = sum(u["completion"]) / n
            avg_tt = sum(u["total"]) / n
            print(f"  Model: {model} (N={n} agent step calls)")
            print(f"    avg prompt_tokens: {avg_pt:.1f}")
            print(f"    avg completion_tokens: {avg_ct:.1f}")
            print(f"    avg total_tokens/call: {avg_tt:.1f}")
            print(f"    min total: {min(u['total'])}, max total: {max(u['total'])}")

if __name__ == "__main__":
    main()
