from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def plot_pareto(rows: list[dict], output_path: Path) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method_name"]].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for method_name, items in grouped.items():
        items = sorted(items, key=lambda item: item["stored_bytes_mean"])
        x = [item["stored_bytes_mean"] for item in items]
        y_sem = [item.get("semantic_recall_mean", item.get("semantic_similarity_mean", 0.0)) for item in items]
        y_scene = [item.get("scene_fidelity_mean", item.get("scene_qa_accuracy_mean", 0.0)) for item in items]
        y_dino = [item.get("dino_similarity_mean", 0.0) for item in items]
        axes[0].plot(x, y_sem, marker="o", label=method_name)
        axes[1].plot(x, y_scene, marker="o", label=method_name)
        axes[2].plot(x, y_dino, marker="o", label=method_name)

    axes[0].set_title("Semantic Metric vs Stored Bytes")
    axes[0].set_xlabel("Actual stored bytes")
    axes[0].set_ylabel("Semantic score")
    axes[1].set_title("Scene QA / Fidelity vs Stored Bytes")
    axes[1].set_xlabel("Actual stored bytes")
    axes[1].set_ylabel("Scene fidelity")
    axes[2].set_title("DINO Similarity vs Stored Bytes")
    axes[2].set_xlabel("Actual stored bytes")
    axes[2].set_ylabel("DINO similarity")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
