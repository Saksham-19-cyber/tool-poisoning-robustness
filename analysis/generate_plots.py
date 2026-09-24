import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_model_size_label(model_name: str) -> str:
    m = model_name.lower()
    if "8b" in m or "3b" in m or "1b" in m or "9b" in m:
        return "Small (~8B)"
    elif "20b" in m or "27b" in m or "17b" in m or "mixtral" in m:
        return "Mid (~20B)"
    elif "70b" in m or "120b" in m:
        return "Large (~70B+)"
    return model_name

def generate_single_turn_plots(results_csv_path: str, output_dir: str):
    if not os.path.exists(results_csv_path):
        print(f"Skipping single-turn plot: {results_csv_path} not found.")
        return

    data = []
    with open(results_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    grouped = defaultdict(lambda: {"total": 0, "attack_succeeded": 0, "task_completed": 0, "brier_sum": 0.0})
    models = sorted(list(set(r["model"] for r in data)))
    conditions = ["clean", "poisoned_explicit", "poisoned_implicit"]

    for r in data:
        key = (r["model"], r["condition"])
        grouped[key]["total"] += 1
        if r["attack_succeeded"].lower() in ("true", "1"):
            grouped[key]["attack_succeeded"] += 1
        if r["task_completed"].lower() in ("true", "1"):
            grouped[key]["task_completed"] += 1
        grouped[key]["brier_sum"] += float(r.get("avg_brier_score", 0.0))

    summary_rows = []
    for m in models:
        for c in conditions:
            g = grouped.get((m, c))
            if g and g["total"] > 0:
                asr = g["attack_succeeded"] / g["total"]
                util = g["task_completed"] / g["total"]
                brier = g["brier_sum"] / g["total"]
                summary_rows.append({
                    "model": m,
                    "size_class": get_model_size_label(m),
                    "condition": c,
                    "total_runs": g["total"],
                    "asr": round(asr, 4),
                    "task_utility": round(util, 4),
                    "mean_brier": round(brier, 4)
                })

    summary_csv = os.path.join(output_dir, "summary_table.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "size_class", "condition", "total_runs", "asr", "task_utility", "mean_brier"])
        writer.writeheader()
        writer.writerows(summary_rows)

    size_order = ["Small (~8B)", "Mid (~20B)", "Large (~70B+)"]
    size_map = {}
    for m in models:
        size_map[get_model_size_label(m)] = m

    ordered_sizes = [s for s in size_order if s in size_map]
    if not ordered_sizes:
        ordered_sizes = [get_model_size_label(m) for m in models]

    plt.figure(figsize=(8, 5), dpi=300)
    for c, marker, color in [("poisoned_explicit", "o-", "#d9534f"), ("poisoned_implicit", "s--", "#f0ad4e"), ("clean", "^-.", "#5cb85c")]:
        vals = []
        for s in ordered_sizes:
            m = size_map[s]
            g = grouped.get((m, c))
            if g and g["total"] > 0:
                vals.append(g["attack_succeeded"] / g["total"] * 100.0)
            else:
                vals.append(0.0)
        label = c.replace("_", " ").title()
        plt.plot(ordered_sizes, vals, marker, label=f"ASR ({label})", color=color, linewidth=2, markersize=8)

    plt.title("Attack Success Rate (ASR) vs. Model Parameter Scale", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Model Parameter Scale", fontsize=11, fontweight="semibold")
    plt.ylabel("Attack Success Rate (%)", fontsize=11, fontweight="semibold")
    plt.ylim(-5, 105)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "asr_vs_model_size.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved ASR plot to {plot_path}")

def generate_multi_turn_plots(results_csv_path: str, output_dir: str):
    if not os.path.exists(results_csv_path):
        print(f"Skipping multi-turn plot: {results_csv_path} not found.")
        return

    data = []
    with open(results_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    turn_brier = defaultdict(lambda: defaultdict(list))
    for r in data:
        m = r["model"]
        turn = int(r["turn"])
        brier = float(r["brier_score"])
        turn_brier[m][turn].append(brier)

    plt.figure(figsize=(8, 5), dpi=300)
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    idx = 0

    for m, turns_dict in turn_brier.items():
        sorted_turns = sorted(turns_dict.keys())
        avg_briers = [sum(turns_dict[t]) / len(turns_dict[t]) for t in sorted_turns]
        size_lbl = get_model_size_label(m)
        color = palette[idx % len(palette)]
        idx += 1
        plt.plot(sorted_turns, avg_briers, "o-", label=f"{size_lbl} ({m})", color=color, linewidth=2, markersize=7)

    plt.title("Confidence Calibration Error (Brier Score) Drift Over Turns", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Conversation Turn (Accumulating Poisoned Context)", fontsize=11, fontweight="semibold")
    plt.ylabel("Calibration Error (Brier Score)", fontsize=11, fontweight="semibold")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "calibration_drift_vs_turn.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Calibration Drift plot to {plot_path}")

def main():
    base_dir = os.path.dirname(__file__)
    output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)

    results_dir = os.path.join(base_dir, "..", "results")
    single_csv = os.path.join(results_dir, "single_turn_results.csv")
    multi_csv = os.path.join(results_dir, "multi_turn_results.csv")

    generate_single_turn_plots(single_csv, output_dir)
    generate_multi_turn_plots(multi_csv, output_dir)

if __name__ == "__main__":
    main()
