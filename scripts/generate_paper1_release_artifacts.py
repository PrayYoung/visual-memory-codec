#!/usr/bin/env python3
"""Generate the Paper 1 release figure from frozen, tracked result tables."""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "visual-memory-codec-matplotlib"))

import matplotlib
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "formal_benchmark" / "canonical_n288" / "aggregate_metrics.csv"
OUTPUT = ROOT / "paper_assets" / "figure_1_rate_qa.png"

METHODS = {
    "webp": ("Rate-controlled WebP", "#1f77b4", "o"),
    "visual_latent_memory": ("Visual latent", "#d62728", "s"),
    "text_only_memory": ("Text only", "#2ca02c", "^"),
    "hybrid_text_visual_latent": ("Hybrid", "#9467bd", "D"),
}

matplotlib.rcParams["svg.hashsalt"] = "paper1-release"


def number(value: str) -> float | None:
    return float(value) if value else None


def main() -> None:
    rows: dict[str, list[dict[str, float]]] = {key: [] for key in METHODS}
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            method = raw["method_name"]
            if method not in METHODS or not raw["scene_qa_accuracy_mean"]:
                continue
            rows[method].append({
                "bytes": float(raw["stored_bytes_mean"]),
                "qa": float(raw["scene_qa_accuracy_mean"]),
                "lo": number(raw["scene_qa_accuracy_ci95_low"]),
                "hi": number(raw["scene_qa_accuracy_ci95_high"]),
            })

    fig, axis = plt.subplots(figsize=(7.2, 4.9))
    for method, values in rows.items():
        label, color, marker = METHODS[method]
        values.sort(key=lambda row: row["bytes"])
        x = [row["bytes"] / 1024 for row in values]
        y = [row["qa"] for row in values]
        lower = [row["qa"] - row["lo"] if row["lo"] is not None else 0 for row in values]
        upper = [row["hi"] - row["qa"] if row["hi"] is not None else 0 for row in values]
        axis.errorbar(x, y, yerr=[lower, upper], marker=marker, color=color,
                      linewidth=2, capsize=2.5, label=label)

    axis.axvline(4, color="#666666", linewidth=0.8, linestyle=":")
    axis.axvline(8, color="#666666", linewidth=0.8, linestyle=":")
    axis.text(4.05, 0.115, "Nominal 4 KB", fontsize=8.5, va="bottom")
    axis.text(8.05, 0.115, "Nominal 8 KB", fontsize=8.5, va="bottom")
    axis.annotate("Latent plateaus at 3.93 KB", (3.928, 0.503), xytext=(4.75, 0.545),
                  arrowprops={"arrowstyle": "-", "color": "#555555"}, fontsize=9)
    axis.set_title("Paper 1: QA accuracy versus actual stored bytes")
    axis.set_xlabel("Mean stored bytes (KB)")
    axis.set_ylabel("Scene-QA accuracy")
    axis.set_xlim(0, 8.8)
    axis.set_ylim(0.08, 0.58)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.subplots_adjust(bottom=0.25, top=0.92)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
