import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_model_size_label(model_name: str) -> str:
    m = model_name.lower()
    if any(s in m for s in ("8b", "3b", "1b", "9b")):
        return "Small (~8B)"
    elif any(s in m for s in ("20b", "27b", "17b", "mixtral")):
        return "Mid (~20B)"
    elif any(s in m for s in ("70b", "120b")):
        return "Large (~70B+)"
    return model_name


def _load_summary_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_detail_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_single_turn_plots(summary_csv: str, detail_csv: str, output_dir: str):
    summary_rows = _load_summary_csv(summary_csv)
    detail_rows = _load_detail_csv(detail_csv)

    if not summary_rows and not detail_rows:
        print("Skipping single-turn plot: no data found.")
        return

    if summary_rows and "asr_ci_lo" in summary_rows[0]:
        rows = summary_rows
        _plot_from_summary(rows, output_dir)
        _write_summary_table(rows, output_dir)
    else:
        rows = _aggregate_from_detail(detail_rows)
        _plot_from_summary(rows, output_dir)
        _write_summary_table(rows, output_dir)


def _aggregate_from_detail(detail_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    from harness.scorer import wilson_ci
    grouped: Dict[Tuple[str, str], Dict] = defaultdict(
        lambda: {"total": 0, "attack": 0, "util": 0, "briers": []}
    )
    for r in detail_rows:
        key = (r["model"], r["condition"])
        grouped[key]["total"] += 1
        if r.get("attack_succeeded", "0") in ("True", "1", "true"):
            grouped[key]["attack"] += 1
        if r.get("task_completed", "0") in ("True", "1", "true"):
            grouped[key]["util"] += 1
        grouped[key]["briers"].append(float(r.get("avg_brier_score", 0.0)))

    rows = []
    for (model, cond), g in grouped.items():
        n = g["total"]
        na = g["attack"]
        nu = g["util"]
        asr_ci = wilson_ci(na, n)
        util_ci = wilson_ci(nu, n)
        mb = sum(g["briers"]) / len(g["briers"]) if g["briers"] else 0.0
        rows.append({
            "model": model,
            "condition": cond,
            "total_runs": n,
            "n_attack": na,
            "n_util": nu,
            "asr": round(na / n, 4),
            "asr_ci_lo": round(asr_ci[0], 4),
            "asr_ci_hi": round(asr_ci[1], 4),
            "task_utility": round(nu / n, 4),
            "utility_ci_lo": round(util_ci[0], 4),
            "utility_ci_hi": round(util_ci[1], 4),
            "mean_brier": round(mb, 4),
            "brier_ci_lo": 0.0,
            "brier_ci_hi": 0.0,
            "ece": 0.0,
        })
    return rows


def _plot_from_summary(rows: List[Dict[str, Any]], output_dir: str):
    models = sorted(set(str(r["model"]) for r in rows))
    size_order = ["Small (~8B)", "Mid (~20B)", "Large (~70B+)"]
    size_map = {get_model_size_label(m): m for m in models}
    ordered_sizes = [s for s in size_order if s in size_map]
    if not ordered_sizes:
        ordered_sizes = [get_model_size_label(m) for m in models]

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
        for s in ordered_sizes:
            m = size_map.get(s)
            row = indexed.get((m, cond)) if m else None
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

        x = list(range(len(ordered_sizes)))

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

    ax.set_xticks(range(len(ordered_sizes)))
    ax.set_xticklabels(ordered_sizes, fontsize=10)
    ax.set_title(
        "Attack Success Rate (ASR) vs. Model Parameter Scale\n(with 95% Wilson confidence intervals)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Model Parameter Scale", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Attack Success Rate (%)", fontsize=11, fontweight="semibold")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "asr_vs_model_size.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved ASR plot -> {plot_path}")


def _write_summary_table(rows: List[Dict[str, Any]], output_dir: str):
    fieldnames = [
        "model", "condition", "total_runs",
        "asr", "asr_ci_lo", "asr_ci_hi",
        "task_utility", "utility_ci_lo", "utility_ci_hi",
        "mean_brier", "brier_ci_lo", "brier_ci_hi",
        "ece",
    ]
    path = os.path.join(output_dir, "summary_table.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary table -> {path}")


def generate_multi_turn_plots(results_csv_path: str, output_dir: str):
    if not os.path.exists(results_csv_path):
        print(f"Skipping multi-turn plot: {results_csv_path} not found.")
        return

    data = _load_detail_csv(results_csv_path)

    turn_brier: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    drift_slopes: Dict[str, List[float]] = defaultdict(list)

    for r in data:
        m = r["model"]
        turn = int(r["turn"])
        brier = float(r["brier_score"])
        turn_brier[m][turn].append(brier)
        if "calibration_drift_slope" in r and r["calibration_drift_slope"]:
            drift_slopes[m].append(float(r["calibration_drift_slope"]))

    if drift_slopes:
        print("\n--- Calibration drift slopes (OLS β, Brier score vs turn) ---")
        for m, slopes in drift_slopes.items():
            avg_slope = sum(slopes) / len(slopes)
            print(f"  {m}: mean drift slope = {avg_slope:+.5f}")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    for idx, (m, turns_dict) in enumerate(sorted(turn_brier.items())):
        sorted_turns = sorted(turns_dict.keys())
        avg_briers = [sum(turns_dict[t]) / len(turns_dict[t]) for t in sorted_turns]
        lo_errs, hi_errs = [], []
        for t in sorted_turns:
            vals = turns_dict[t]
            n = len(vals)
            mean = sum(vals) / n
            se = (sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5 / (n ** 0.5)
            lo_errs.append(1.96 * se)
            hi_errs.append(1.96 * se)

        size_lbl = get_model_size_label(m)
        color = palette[idx % len(palette)]
        ax.errorbar(
            sorted_turns, avg_briers,
            yerr=[lo_errs, hi_errs],
            fmt="o-",
            label=f"{size_lbl} ({m})",
            color=color,
            linewidth=2,
            markersize=7,
            capsize=4,
            elinewidth=1.5,
            alpha=0.9,
        )

    ax.set_title(
        "Confidence Calibration Error (Brier Score) Drift Over Turns\n"
        "(poisoned context accumulates; error bars = ±1.96 SE)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Conversation Turn (Accumulating Poisoned Context)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Calibration Error (Brier Score)", fontsize=11, fontweight="semibold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "calibration_drift_vs_turn.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Calibration Drift plot -> {plot_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)

    results_dir = os.path.join(base_dir, "..", "results")
    summary_csv = os.path.join(base_dir, "summary_table.csv")
    single_detail_csv = os.path.join(results_dir, "single_turn_results.csv")
    multi_csv = os.path.join(results_dir, "multi_turn_results.csv")

    generate_single_turn_plots(summary_csv, single_detail_csv, output_dir)
    generate_multi_turn_plots(multi_csv, output_dir)


if __name__ == "__main__":
    main()
