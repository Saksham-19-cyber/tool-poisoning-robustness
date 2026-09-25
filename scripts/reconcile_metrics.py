import os
import json
import glob
import re
from collections import defaultdict

def main():
    # 1. Map transcripts to runs
    # Pattern: {slug}_{condition}_{task_id}_trial{trial}_step{step}.json
    transcript_files = glob.glob("results/raw_transcripts/*_trial*_step*.json")
    
    runs_steps = defaultdict(list)
    step_tokens_by_model = defaultdict(list)
    
    for f in transcript_files:
        basename = os.path.basename(f)
        m = re.match(r"^(.*)_(clean|poisoned_explicit|poisoned_implicit)_(task_\d+)_trial(\d+)_step(\d+)\.json$", basename)
        if m:
            slug, cond, task_id, trial, step = m.groups()
            run_key = (slug, cond, task_id, int(trial))
            with open(f, "r", encoding="utf-8") as jf:
                d = json.load(jf)
            runs_steps[run_key].append((int(step), d))
            
            model = d.get("model", slug)
            usage = d.get("usage", {})
            tt = usage.get("total_tokens", 0)
            if tt > 0:
                step_tokens_by_model[model].append(tt)
    
    print(f"Total mapped runs from transcripts: {len(runs_steps)}")
    
    # Analyze agent steps per run
    model_steps = defaultdict(list)
    for (slug, cond, task_id, trial), steps in runs_steps.items():
        model_steps[slug].append(len(steps))
        
    for slug, steps_list in sorted(model_steps.items()):
        avg_s = sum(steps_list) / len(steps_list)
        print(f"  {slug}: avg agent steps/run = {avg_s:.2f} (min={min(steps_list)}, max={max(steps_list)}, N={len(steps_list)})")

    # 2. Analyze tool calls per run from single_turn_results_live.json
    with open("results/single_turn_results_live.json", "r", encoding="utf-8") as f:
        records = json.load(f)
        
    model_tool_calls = defaultdict(list)
    for r in records:
        model = r.get("model")
        tc = r.get("num_tool_calls", 0)
        model_tool_calls[model].append(tc)
        
    print("\n--- Tool calls per run from single_turn_results_live.json ---")
    for model, tcs in sorted(model_tool_calls.items()):
        avg_tc = sum(tcs) / len(tcs)
        print(f"  {model}: avg tool_calls/run = {avg_tc:.2f} (min={min(tcs)}, max={max(tcs)}, N={len(tcs)})")

    # 3. Calculate total API requests per run
    # Total API requests = Agent Steps + Tool Calls (elicitations)
    print("\n--- Total API Requests per Run ---")
    # Let's match by slug
    slug_map = {
        "openai_gpt-oss-120b": "openai/gpt-oss-120b",
        "openai_gpt-oss-20b": "openai/gpt-oss-20b",
        "qwen_qwen3.8-27b": "qwen/qwen3.8-27b",
    }
    
    for slug, model in slug_map.items():
        s_list = model_steps.get(slug, [])
        tc_list = model_tool_calls.get(model, [])
        avg_s = sum(s_list) / len(s_list) if s_list else 1.8
        avg_tc = sum(tc_list) / len(tc_list) if tc_list else 2.0
        avg_reqs = avg_s + avg_tc
        print(f"  {model}:")
        print(f"    Avg Agent Steps: {avg_s:.2f}")
        print(f"    Avg Elicitations (Tool Calls): {avg_tc:.2f}")
        print(f"    Total API Requests/run: {avg_reqs:.2f}")

    # 4. Token reconciliation
    print("\n--- Token Consumption per Run & per Request ---")
    for model, tokens in sorted(step_tokens_by_model.items()):
        avg_agent_tok = sum(tokens) / len(tokens) if tokens else 0
        # Elicitation tokens: prompt is ~180 tokens, completion is ~40 tokens -> ~220 tokens
        avg_elicit_tok = 220.0
        # Find corresponding avg_s and avg_tc
        slug = model.replace("/", "_")
        avg_s = sum(model_steps.get(slug, [1.8])) / len(model_steps.get(slug, [1.8]))
        avg_tc = sum(model_tool_calls.get(model, [2.0])) / len(model_tool_calls.get(model, [2.0]))
        total_tok_run = (avg_s * avg_agent_tok) + (avg_tc * avg_elicit_tok)
        avg_tok_req = total_tok_run / (avg_s + avg_tc)
        print(f"  {model}:")
        print(f"    Avg tokens per agent step: {avg_agent_tok:.1f}")
        print(f"    Avg tokens per elicitation call: ~{avg_elicit_tok:.1f}")
        print(f"    Weighted avg tokens per API request: {avg_tok_req:.1f}")
        print(f"    Calculated total tokens per RUN: {total_tok_run:.1f}")

if __name__ == "__main__":
    main()
