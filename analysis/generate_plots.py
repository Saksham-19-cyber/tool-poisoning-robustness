import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_model_display_name(model_name: str) -> str:
    m = model_name.lower()
    if "120b" in m:
        return "GPT-OSS-120B (120B)"
    elif "27b" in m:
        return "Qwen3.8-27B (27B)"
    elif "20b" in m:
        return "GPT-OSS-20B (20B)"
    return model_name


def get_model_param_order(model_name: str) -> int:
    m = model_name.lower()
    if "120b" in m:
        return 120
    elif "27b" in m:
        return 27
    elif "20b" in m:
        return 20
    return 50


def _load_json_data(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_data(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wilson_ci(k: int, n: int) -> Tuple[float, float]:
    import math
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p_hat = k / n
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_aggregate_summary(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict] = defaultdict(
        lambda: {"total": 0, "attack": 0, "util": 0, "briers": [], "steps": []}
    )
    for r in detail_rows:
        key = (r["model"], r["condition"])
        grouped[key]["total"] += 1
        if r.get("attack_succeeded") in (True, "True", 1, "1", "true"):
            grouped[key]["attack"] += 1
        if r.get("task_completed") in (True, "True", 1, "1", "true"):
            grouped[key]["util"] += 1
        grouped[key]["briers"].append(float(r.get("avg_brier_score", 0.0)))
        if "step_evals" in r:
            grouped[key]["steps"].extend(r["step_evals"])

    rows = []
    models = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    conditions = ["clean", "poisoned_explicit", "poisoned_implicit"]

    for m in models:
        for cond in conditions:
            if (m, cond) not in grouped:
                continue
            g = grouped[(m, cond)]
            n = g["total"]
            na = g["attack"]
            nu = g["util"]
            asr_lo, asr_hi = wilson_ci(na, n)
            util_lo, util_hi = wilson_ci(nu, n)
            mb = sum(g["briers"]) / len(g["briers"]) if g["briers"] else 0.0
            
            # ECE calculation
            steps = g["steps"]
            if steps:
                import numpy as np
                confidences = np.array([e["confidence_prob"] for e in steps])
                correctness = np.array([1.0 if e.get("is_correct", False) else 0.0 for e in steps])
                bin_boundaries = np.linspace(0, 1, 11)
                ece = 0.0
                n_s = len(confidences)
                for i in range(10):
                    b_low, b_high = bin_boundaries[i], bin_boundaries[i+1]
                    in_bin = (confidences > b_low) & (confidences <= b_high) if i > 0 else (confidences >= b_low) & (confidences <= b_high)
                    bin_count = np.sum(in_bin)
                    if bin_count > 0:
                        avg_conf = np.mean(confidences[in_bin])
                        avg_acc = np.mean(correctness[in_bin])
                        ece += (bin_count / n_s) * abs(avg_acc - avg_conf)
            else:
                ece = 0.0

            rows.append({
                "model": m,
                "condition": cond,
                "total_runs": n,
                "asr": round(na / n, 4),
                "asr_ci_lo": round(asr_lo, 4),
                "asr_ci_hi": round(asr_hi, 4),
                "task_utility": round(nu / n, 4),
                "utility_ci_lo": round(util_lo, 4),
                "utility_ci_hi": round(util_hi, 4),
                "mean_brier": round(mb, 4),
                "ece": round(float(ece), 4),
            })
    return rows


def plot_asr_from_summary(rows: List[Dict[str, Any]], output_dir: str):
    models = sorted(set(str(r["model"]) for r in rows), key=get_model_param_order)
    model_labels = [get_model_display_name(m) for m in models]

    indexed: Dict[Tuple[str, str], Dict] = {
        (str(r["model"]), str(r["condition"])): r for r in rows
    }

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    configs = [
        ("poisoned_explicit", "o", "-",  "#d9534f", "Explicit Poisoning (ASR)"),
        ("poisoned_implicit", "s", "--", "#f0ad4e", "Implicit Poisoning (ASR)"),
        ("clean",             "^", "-.", "#5cb85c", "Clean Baseline (ASR)"),
    ]

    for cond, marker, linestyle, color, label in configs:
        vals, lo_errs, hi_errs = [], [], []
        for m in models:
            row = indexed.get((m, cond))
            if row:
                asr = float(row.get("asr", 0.0))
                lo = float(row.get("asr_ci_lo", asr))
                hi = float(row.get("asr_ci_hi", asr))
                vals.append(asr * 100)
                lo_errs.append((asr - lo) * 100)
                hi_errs.append((hi - asr) * 100)
            else:
                vals.append(0.0)
                lo_errs.append(0.0)
                hi_errs.append(0.0)

        x = list(range(len(models)))

        ax.errorbar(
            x, vals,
            yerr=[lo_errs, hi_errs],
            fmt=marker,
            linestyle=linestyle,
            label=label,
            color=color,
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.9,
        )

    # Annotate underpowered implicit points
    # 20B implicit is index 0
    # Qwen implicit is index 1
    ax.annotate(
        "N=11 (Wide CI)",
        xy=(0, 0.0), xytext=(0, -9),
        arrowprops=dict(facecolor='#f0ad4e', edgecolor='#d58512', arrowstyle='->', lw=1.2),
        fontsize=8.5, fontweight='bold', color='#8a6d3b', ha='center',
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fcf8e3", edgecolor="#faebcc")
    )
    ax.annotate(
        "N=7 (Wide CI)",
        xy=(1, 0.0), xytext=(1, -9),
        arrowprops=dict(facecolor='#f0ad4e', edgecolor='#d58512', arrowstyle='->', lw=1.2),
        fontsize=8.5, fontweight='bold', color='#8a6d3b', ha='center',
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fcf8e3", edgecolor="#faebcc")
    )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, fontsize=10.5, fontweight="semibold")
    ax.set_title(
        "Attack Success Rate (ASR) vs. Model Parameter Scale\n(Live Groq API Benchmark, N=256, 95% Wilson CIs)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Evaluated Model Scale / Architecture", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Attack Success Rate (%)", fontsize=11, fontweight="semibold")
    ax.set_ylim(-15, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "asr_vs_model_size.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved ASR plot -> {plot_path}")


def write_summary_table(rows: List[Dict[str, Any]], output_dir: str):
    fieldnames = [
        "model", "condition", "total_runs",
        "asr", "asr_ci_lo", "asr_ci_hi",
        "task_utility", "utility_ci_lo", "utility_ci_hi",
        "mean_brier", "ece"
    ]
    path = os.path.join(output_dir, "summary_table.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary table -> {path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)

    results_dir = os.path.join(base_dir, "..", "results")
    live_json = os.path.join(results_dir, "single_turn_results_live.json")
    
    if os.path.exists(live_json):
        print(f"Loading live data from {live_json}")
        detail_rows = _load_json_data(live_json)
    else:
        live_csv = os.path.join(results_dir, "single_turn_results_live.csv")
        print(f"Loading live data from {live_csv}")
        detail_rows = _load_csv_data(live_csv)

    rows = compute_aggregate_summary(detail_rows)
    plot_asr_from_summary(rows, output_dir)
    write_summary_table(rows, output_dir)


if __name__ == "__main__":
    main()
