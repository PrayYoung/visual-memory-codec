"""Shared utilities for the frozen natural-image benchmark scripts."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from visual_memory_benchmark.codecs.image_memory import (
    RATE_CONTROLLED_QUALITIES,
    RATE_CONTROLLED_RESOLUTIONS,
    PillowImageCodec,
)
from visual_memory_benchmark.codecs.real_text import RealTextCodec
from visual_memory_benchmark.codecs.real_visual_latent import RealVisualLatentCodec
from visual_memory_benchmark.types import EncodedArtifact


BUDGETS = [256, 512, 1024, 2048, 4096, 8192]


class HybridReal:
    def __init__(self, text: RealTextCodec, latent: RealVisualLatentCodec) -> None:
        self.text, self.latent = text, latent

    def encode(self, sample, budget: int) -> EncodedArtifact:
        if budget < 6:
            raise ValueError("hybrid budget must accommodate its 4-byte container header")
        text_budget = (budget - 4) // 2
        latent_budget = budget - 4 - text_budget
        text = self.text.encode(sample, text_budget)
        latent = self.latent.encode(sample, latent_budget)
        payload = len(text.payload).to_bytes(4, "big") + text.payload + latent.payload
        if len(payload) > budget:
            raise RuntimeError("hybrid stored bytes exceed budget")
        return EncodedArtifact("hybrid_text_visual_latent", payload, {"text": text.aux["text"], "latent": latent})

    def decode(self, artifact):
        return self.latent.decode(artifact.aux["latent"])


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def run_codec_preflight(samples, out: Path) -> None:
    """CPU-only feasibility gate for the conventional image-codec arms."""
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    codecs = {
        "webp_rate_controlled": PillowImageCodec("webp_rate_controlled", 256, "WEBP"),
        "webp_fixed_resolution_diagnostic": PillowImageCodec(
            "webp_fixed_resolution_diagnostic", 256, "WEBP", resolutions=(256,)
        ),
        "avif_rate_controlled": PillowImageCodec("avif_rate_controlled", 256, "AVIF"),
    }
    availability: dict[str, dict] = {}
    for name, codec in codecs.items():
        try:
            for sample in samples:
                for budget in BUDGETS:
                    try:
                        artifact = codec.encode(sample, budget)
                        rows.append({"method_name": name, "sample_id": sample.sample_id, "budget_bytes": budget,
                                     "status": "feasible", "actual_bytes": artifact.stored_bytes,
                                     "encoded_resolution": artifact.aux["encoded_resolution"], "quality": artifact.aux["quality"]})
                    except ValueError as error:
                        rows.append({"method_name": name, "sample_id": sample.sample_id, "budget_bytes": budget,
                                     "status": "infeasible", "reason": str(error)})
            availability[name] = {"available": True}
        except Exception as error:
            availability[name] = {"available": False, "reason": str(error)}
    write_rows(out / "codec_feasibility_per_image.csv", rows)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method_name"], row["budget_bytes"])].append(row)
    summary = []
    for (method, budget), group in sorted(grouped.items()):
        feasible = [row for row in group if row["status"] == "feasible"]
        summary.append({"method_name": method, "budget_bytes": budget, "n_images": len(group),
                        "n_feasible": len(feasible), "n_infeasible": len(group) - len(feasible),
                        "actual_bytes_mean": sum(row["actual_bytes"] for row in feasible) / len(feasible) if feasible else None})
    write_rows(out / "codec_feasibility_summary.csv", summary)
    (out / "codec_policy.json").write_text(json.dumps({
        "selection": "first valid candidate in predeclared descending resolution then descending quality grid",
        "resolutions": RATE_CONTROLLED_RESOLUTIONS,
        "qualities": RATE_CONTROLLED_QUALITIES,
        "selection_inputs": "actual stored bytes only",
        "fixed_resolution_webp_diagnostic": 256,
        "availability": availability,
    }, indent=2))
