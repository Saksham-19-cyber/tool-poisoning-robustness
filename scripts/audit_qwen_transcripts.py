import os
import json
import glob

def main():
    with open("results/single_turn_results_live.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    with open("tasks/tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)
    task_map = {t["id"]: t for t in tasks}

    with open("tool_registry/poisoned_explicit.json", "r", encoding="utf-8") as tf:
        schemas = json.load(tf)
    schema_map = {s["function"]["name"]: s["function"] for s in schemas}

    qwen_pe = [r for r in records if r.get("model") == "qwen/qwen3.8-27b" and r.get("condition") == "poisoned_explicit"]
    print(f"Total Qwen PE runs in single_turn_results_live.json: {len(qwen_pe)}")

    for r in qwen_pe:
        tid = r.get("task_id")
        trial = r.get("trial")
        t = task_map.get(tid, {})
        malicious_tool = t.get("malicious_tool")
        target_sig = t.get("target_signature")
        expected_tools = t.get("expected_tools", [])
        print(f"\n{'='*70}")
        print(f"RUN: {tid} trial {trial} | Comp={r.get('task_completed')} | ASR={r.get('attack_succeeded')}")
        print(f"  User Prompt: {t.get('prompt')}")
        print(f"  Expected Tools: {expected_tools}")
        print(f"  Malicious Tool: {malicious_tool} (target_sig: '{target_sig}')")
        
        # Check the description of the expected tool in poisoned_explicit
        for et in expected_tools:
            fn_def = schema_map.get(et, {})
            desc = fn_def.get("description", "")
            print(f"  Tool Schema Description for '{et}':")
            print(f"    {desc}")
        
        # Check raw transcripts for this run
        pattern = f"results/raw_transcripts/qwen_qwen3.8-27b_poisoned_explicit_{tid}_trial{trial}_step*.json"
        matching_files = sorted(glob.glob(pattern))
        if not matching_files:
            pattern_legacy = f"results/raw_transcripts/qwen_qwen3.8-27b_poisoned_explicit_{tid}_step*.json"
            matching_files = sorted(glob.glob(pattern_legacy))
        
        print(f"  Raw transcript step files ({len(matching_files)}):")
        for step_file in matching_files:
            with open(step_file, "r", encoding="utf-8") as sf:
                sd = json.load(sf)
            choices = sd.get("choices", [])
            msg = choices[0].get("message", {}) if choices else {}
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls") or []
            print(f"    File: {os.path.basename(step_file)}")
            print(f"      message.content: {repr(content)}")
            print(f"      parsed tool_calls ({len(tool_calls)}):")
            for tc in tool_calls:
                fn = tc.get("function", {})
                print(f"        name: {fn.get('name')}")
                print(f"        arguments: {fn.get('arguments')}")

if __name__ == "__main__":
    main()
