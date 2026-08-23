#!/usr/bin/env python3
"""Frozen formal runner using the staged GQA/Visual Genome manifest."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from PIL import Image

from benchmark_common import BUDGETS, HybridReal, run_codec_preflight, write_rows
from visual_memory_benchmark.codecs.image_memory import OriginalRawCodec, PillowImageCodec
from visual_memory_benchmark.codecs.real_text import RealTextCodec
from visual_memory_benchmark.codecs.real_visual_latent import RealVisualLatentCodec
from visual_memory_benchmark.eval.natural_metrics import NaturalMetricEvaluator
from visual_memory_benchmark.models.hf_adapters import Qwen25VlMemoryQa
from visual_memory_benchmark.types import SceneSample


SEED = 20260820


def load_manifest(path: Path) -> tuple[list[SceneSample], dict[str, list[dict]], dict]:
    data = json.loads(path.read_text())
    if data["n_images"] != 288 or data["n_questions"] != 1152:
        raise RuntimeError("Formal run requires the frozen N=288 / 1,152-question manifest")
    samples, qa = [], {}
    for item in data["images"]:
        image_path = Path(item["image_relpath"])
        if not image_path.exists():
            raise RuntimeError(f"Missing staged image {image_path}")
        image = Image.open(image_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        sample_id = f"gqa_{item['image_id']}"
        samples.append(SceneSample(sample_id=sample_id, image=image, objects=None,
                                   metadata={"gqa_image_id": item["image_id"]}, source_path=str(image_path)))
        qa[sample_id] = item["qa"]
    counts = defaultdict(int)
    for questions in qa.values():
        if len(questions) != 4: raise RuntimeError("Each formal image must have four frozen questions")
        for question in questions: counts[question["group"]] += 1
    if dict(counts) != data["group_question_counts"]:
        raise RuntimeError("Manifest functional-group count mismatch")
    return samples, qa, data


def normalize(answer: str, expected: str) -> str:
    value = re.sub(r"[^a-z0-9 ]+", " ", answer.lower()).strip()
    value = re.sub(r"\s+", " ", value)
    expected = re.sub(r"[^a-z0-9 ]+", " ", expected.lower()).strip()
    words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6"}
    if expected.isdigit():
        match = re.search(r"\b[0-9]+\b", value)
        if match: return match.group(0)
        return words.get(value, value)
    if expected in {"yes", "no"}:
        match = re.search(r"\b(yes|no)\b", value)
        return match.group(1) if match else value
    prefixes = ("the answer is ", "answer is ", "it is ", "it s ", "a ", "an ", "the ")
    for prefix in prefixes:
        if value.startswith(prefix): value = value[len(prefix):]
    return expected if value == expected or value.endswith(" " + expected) else value


def bootstrap(values: list[float], rng: random.Random) -> tuple[float | None, float | None]:
    if not values: return None, None
    draws = sorted(sum(values[rng.randrange(len(values))] for _ in range(len(values))) / len(values) for _ in range(10_000))
    return draws[249], draws[9749]


def summarize_groups(answers: list[dict], output: Path) -> None:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in answers:
        for answer in row["answers"]:
            grouped[(row["method_name"], row["budget_bytes"], answer["group"])][row["sample_id"]].append(float(answer["correct"]))
    rng = random.Random(SEED)
    rows = []
    for (method, budget, group), by_image in sorted(grouped.items()):
        image_ids = sorted(by_image); vals = [sum(by_image[i])/len(by_image[i]) for i in image_ids]
        lo, hi = bootstrap(vals, rng)
        rows.append({"method_name": method, "budget_bytes": budget, "functional_group": group,
                     "n_images": len(image_ids), "n_questions": sum(len(v) for v in by_image.values()),
                     "qa_accuracy_mean": sum(v for xs in by_image.values() for v in xs) / sum(len(xs) for xs in by_image.values()),
                     "image_bootstrap_ci95_low": lo, "image_bootstrap_ci95_high": hi})
    write_rows(output / "functional_group_qa.csv", rows)
    paired_rows = []
    for budget, group in sorted({(budget, group) for _, budget, group in grouped}):
        webp = grouped.get(("webp", budget, group), {})
        for method in sorted({method for method, arm_budget, arm_group in grouped if arm_budget == budget and arm_group == group}):
            current = grouped[(method, budget, group)]
            image_ids = sorted(set(webp) & set(current))
            if not image_ids: continue
            point = (sum(v for image_id in image_ids for v in current[image_id]) / sum(len(current[i]) for i in image_ids)
                     - sum(v for image_id in image_ids for v in webp[image_id]) / sum(len(webp[i]) for i in image_ids))
            rng = random.Random(f"{SEED}:{method}:{budget}:{group}")
            draws = []
            for _ in range(10_000):
                sampled = [image_ids[rng.randrange(len(image_ids))] for _ in image_ids]
                a = [v for image_id in sampled for v in current[image_id]]
                b = [v for image_id in sampled for v in webp[image_id]]
                draws.append(sum(a)/len(a) - sum(b)/len(b))
            draws.sort(); lo, hi = draws[249], draws[9749]
            paired_rows.append({"method_name": method, "budget_bytes": budget, "functional_group": group,
                                "reference_method": "webp", "n_paired_images": len(image_ids),
                                "difference_vs_webp": point, "paired_image_bootstrap_ci95_low": lo,
                                "paired_image_bootstrap_ci95_high": hi})
    write_rows(output / "functional_group_pairwise_vs_webp.csv", paired_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/gqa/formal_n288_manifest.json")
    parser.add_argument("--output", default="outputs/formal_n288")
    parser.add_argument("--codec-preflight", action="store_true")
    args = parser.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    samples, frozen, manifest = load_manifest(Path(args.manifest))
    (out / "frozen_qa.json").write_text(json.dumps(frozen, indent=2))
    (out / "manifest_provenance.json").write_text(json.dumps({k: manifest[k] for k in manifest if k != "images"}, indent=2))
    if args.codec_preflight:
        run_codec_preflight(samples, out); return
    (out / "artifacts").mkdir(exist_ok=True); (out / "reconstructions").mkdir(exist_ok=True)
    text = RealTextCodec("text_only_memory", 256); latent = RealVisualLatentCodec("visual_latent_memory", 256)
    methods = {"text_only_memory": text, "visual_latent_memory": latent,
               "hybrid_text_visual_latent": HybridReal(text, latent), "webp": PillowImageCodec("webp", 256, "WEBP"),
               "webp_fixed_resolution_diagnostic": PillowImageCodec("webp_fixed_resolution_diagnostic", 256, "WEBP", resolutions=(256,)),
               "original_raw_oracle": OriginalRawCodec("original_raw_oracle", 256)}
    try:
        avif = PillowImageCodec("avif", 256, "AVIF"); avif.encode(samples[0], 8192); methods["avif"] = avif
    except Exception as exc:
        (out / "avif_status.json").write_text(json.dumps({"included": False, "reason": str(exc)}, indent=2))
    qa, diagnostics, rows, answers = Qwen25VlMemoryQa(), NaturalMetricEvaluator(), [], []
    for name, codec in methods.items():
        for budget in BUDGETS:
            for sample in samples:
                start = perf_counter()
                try: artifact = codec.encode(sample, budget)
                except ValueError as exc:
                    rows.append({"sample_id": sample.sample_id, "method_name": name, "budget_bytes": budget, "status": "infeasible", "reason": str(exc)}); continue
                path = out / "artifacts" / name / str(budget) / f"{sample.sample_id}.bin"; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(artifact.payload)
                actual = path.stat().st_size
                if actual != artifact.stored_bytes: raise RuntimeError("artifact accounting mismatch")
                stored_text = artifact.aux.get("text"); image = None if name == "text_only_memory" else codec.decode(artifact)
                if image is not None:
                    recon = out / "reconstructions" / name / str(budget) / f"{sample.sample_id}.png"; recon.parent.mkdir(parents=True, exist_ok=True); image.save(recon)
                qrows = []
                for question in frozen[sample.sample_id]:
                    raw = qa.answer(question["question"], text=stored_text, image=image)
                    pred = normalize(raw, question["answer"]); qrows.append({**question, "prediction_raw": raw, "prediction": pred, "correct": pred == question["answer"]})
                metrics = diagnostics.evaluate(sample.image, image) if image is not None else {}
                rows.append({"sample_id": sample.sample_id, "method_name": name, "budget_bytes": budget, "status": "ok", "stored_bytes": actual, "artifact_file_bytes": actual, "budget_utilization": actual/budget, "scene_qa_accuracy": sum(q["correct"] for q in qrows)/len(qrows), "encode_seconds": perf_counter()-start, **metrics})
                answers.append({"sample_id": sample.sample_id, "method_name": name, "budget_bytes": budget, "answers": qrows})
    write_rows(out / "per_scene_metrics.csv", rows)
    (out / "qa_predictions.json").write_text(json.dumps(answers, indent=2))
    grouped = defaultdict(list)
    for row in rows: grouped[(row["method_name"], row["budget_bytes"])].append(row)
    rng = random.Random(SEED); aggregate = []
    for (method, budget), group in sorted(grouped.items()):
        feasible = [row for row in group if row["status"] == "ok"]
        scores = [row["scene_qa_accuracy"] for row in feasible]
        lo, hi = bootstrap(scores, rng)
        record = {"method_name": method, "budget_bytes": budget, "n_attempted": len(group), "n_feasible": len(feasible),
                  "n_infeasible": len(group)-len(feasible), "stored_bytes_mean": sum(row["stored_bytes"] for row in feasible)/len(feasible) if feasible else None,
                  "scene_qa_accuracy_mean": sum(scores)/len(scores) if scores else None, "scene_qa_accuracy_ci95_low": lo, "scene_qa_accuracy_ci95_high": hi}
        for metric in ("semantic_similarity", "dino_similarity", "psnr", "ssim_like"):
            values = [row[metric] for row in feasible if metric in row]
            record[f"{metric}_mean"] = sum(values)/len(values) if values else None
        aggregate.append(record)
    write_rows(out / "aggregate_metrics.csv", aggregate)
    write_rows(out / "infeasible_arms.csv", [row for row in rows if row["status"] == "infeasible"])
    (out / "protocol.json").write_text(json.dumps({"dataset": manifest["dataset"], "manifest_sha256": __import__("hashlib").sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "qa": "Frozen GQA/Visual Genome ground-truth manifest; evaluator receives only stored text and/or decoded visual memory after encoding.",
        "bootstrap": {"resamples": 10000, "seed": SEED, "unit": "image"}, "functional_groups": manifest["group_question_counts"]}, indent=2))
    summarize_groups(answers, out)


if __name__ == "__main__": main()
