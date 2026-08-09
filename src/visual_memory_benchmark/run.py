from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
from time import perf_counter

from visual_memory_benchmark.codecs.hybrid import HybridCodec
from visual_memory_benchmark.codecs.real_text import RealTextCodec
from visual_memory_benchmark.codecs.real_visual_latent import RealVisualLatentCodec
from visual_memory_benchmark.codecs.text_only import TextOnlyCodec
from visual_memory_benchmark.codecs.visual_latent import VisualLatentCodec
from visual_memory_benchmark.config import ExperimentConfig, MethodConfig
from visual_memory_benchmark.data.coco_subset import CocoSubsetDataset
from visual_memory_benchmark.data.synthetic_shapes import SyntheticShapesDataset
from visual_memory_benchmark.eval.coco_qa import answer_coco_qa, generate_coco_qa
from visual_memory_benchmark.eval.natural_metrics import NaturalMetricEvaluator
from visual_memory_benchmark.models.hf_adapters import DetrObjectDetector
from visual_memory_benchmark.eval.scene_metrics import (
    answer_scene_qa,
    generate_scene_qa,
    psnr,
    scene_fidelity,
    semantic_recall,
    ssim_like,
)
from visual_memory_benchmark.utils.comparison import write_comparison_html
from visual_memory_benchmark.utils.io import ensure_dir, write_csv, write_json
from visual_memory_benchmark.utils.plots import plot_pareto


def build_codec(method: MethodConfig, image_size: int):
    if method.kind == "text_only":
        return TextOnlyCodec(method_name=method.name, image_size=image_size, **method.params)
    if method.kind == "visual_latent_only":
        return VisualLatentCodec(method_name=method.name, image_size=image_size, **method.params)
    if method.kind == "hybrid":
        return HybridCodec(method_name=method.name, image_size=image_size, **method.params)
    if method.kind == "text_only_real":
        return RealTextCodec(method_name=method.name, image_size=image_size, **method.params)
    if method.kind == "visual_latent_real":
        return RealVisualLatentCodec(method_name=method.name, image_size=image_size, **method.params)
    raise ValueError(f"Unknown method kind: {method.kind}")


def build_dataset(config: ExperimentConfig):
    dataset_type = config.dataset.type
    params = config.dataset.params
    if dataset_type == "synthetic_shapes":
        return SyntheticShapesDataset(
            num_samples=params["num_samples"],
            image_size=params["image_size"],
            min_objects=params["min_objects"],
            max_objects=params["max_objects"],
            seed=params["seed"],
        )
    if dataset_type == "coco_subset":
        return CocoSubsetDataset(
            images_dir=params["images_dir"],
            annotations_path=params["annotations_path"],
            image_size=params["image_size"],
            num_samples=params["num_samples"],
            min_annotations=params.get("min_annotations", 2),
            require_multiple_categories=params.get("require_multiple_categories", True),
            include_image_ids=params.get("include_image_ids"),
        )
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def run_synthetic(config: ExperimentConfig, run_dir: Path) -> None:
    recon_dir = ensure_dir(run_dir / "reconstructions")
    qa_dir = ensure_dir(run_dir / "qa")
    dataset = SyntheticShapesDataset(
        num_samples=config.dataset.params["num_samples"],
        image_size=config.dataset.params["image_size"],
        min_objects=config.dataset.params["min_objects"],
        max_objects=config.dataset.params["max_objects"],
        seed=config.dataset.params["seed"],
    )
    samples = dataset.iter_samples()

    per_scene_rows: list[dict] = []
    aggregate_buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)

    for method in config.methods:
        codec = build_codec(method, config.dataset.params["image_size"])
        for budget_bytes in config.budgets_bytes:
            method_budget_dir = ensure_dir(recon_dir / method.name / str(budget_bytes))
            qa_rows: list[dict] = []
            for sample in samples:
                encode_start = perf_counter()
                artifact = codec.encode(sample, budget_bytes)
                encode_seconds = perf_counter() - encode_start

                decode_start = perf_counter()
                reconstruction = codec.decode(artifact)
                decode_seconds = perf_counter() - decode_start

                reconstruction.save(method_budget_dir / f"{sample.sample_id}.png")

                qa_pairs = generate_scene_qa(sample)
                qa_score = answer_scene_qa(reconstruction, sample, qa_pairs)
                qa_rows.append({"sample_id": sample.sample_id, "qa_pairs": qa_pairs})

                row = {
                    "sample_id": sample.sample_id,
                    "method_name": method.name,
                    "budget_bytes": budget_bytes,
                    "stored_bytes": artifact.stored_bytes,
                    "budget_utilization": artifact.stored_bytes / budget_bytes,
                    "semantic_recall": semantic_recall(sample, reconstruction),
                    "scene_fidelity": scene_fidelity(sample, reconstruction),
                    "scene_qa_accuracy": qa_score,
                    "psnr": psnr(sample.image, reconstruction),
                    "ssim_like": ssim_like(sample.image, reconstruction),
                    "encode_seconds": encode_seconds,
                    "decode_seconds": decode_seconds,
                }
                per_scene_rows.append(row)
                aggregate_buckets[(method.name, budget_bytes)].append(row)
            write_json(qa_dir / f"{method.name}_{budget_bytes}.json", qa_rows)

    aggregate_rows: list[dict] = []
    for (method_name, budget_bytes), rows in sorted(aggregate_buckets.items()):
        count = len(rows)
        aggregate_rows.append(
            {
                "method_name": method_name,
                "budget_bytes": budget_bytes,
                "stored_bytes_mean": sum(row["stored_bytes"] for row in rows) / count,
                "budget_utilization_mean": sum(row["budget_utilization"] for row in rows) / count,
                "semantic_recall_mean": sum(row["semantic_recall"] for row in rows) / count,
                "scene_fidelity_mean": sum(row["scene_fidelity"] for row in rows) / count,
                "scene_qa_accuracy_mean": sum(row["scene_qa_accuracy"] for row in rows) / count,
                "psnr_mean": sum(row["psnr"] for row in rows) / count,
                "ssim_like_mean": sum(row["ssim_like"] for row in rows) / count,
                "encode_seconds_mean": sum(row["encode_seconds"] for row in rows) / count,
                "decode_seconds_mean": sum(row["decode_seconds"] for row in rows) / count,
            }
        )

    write_csv(run_dir / "per_scene_metrics.csv", per_scene_rows)
    write_csv(run_dir / "aggregate_metrics.csv", aggregate_rows)
    plot_pareto(aggregate_rows, run_dir / "pareto_curves.png")


def run_natural(config: ExperimentConfig, run_dir: Path) -> None:
    recon_dir = ensure_dir(run_dir / "reconstructions")
    qa_dir = ensure_dir(run_dir / "qa")
    artifact_dir = ensure_dir(run_dir / "artifacts")
    original_dir = ensure_dir(run_dir / "originals")
    diagnostics_dir = ensure_dir(run_dir / "diagnostics")

    dataset = build_dataset(config)
    samples = dataset.iter_samples()
    image_size = config.dataset.params["image_size"]

    eval_params = config.evaluation.params if config.evaluation else {}
    metric_evaluator = NaturalMetricEvaluator(
        semantic_model_name=eval_params.get("semantic_model_name", "google/siglip-base-patch16-224"),
        dino_model_name=eval_params.get("dino_model_name", "facebook/dinov2-base"),
    )
    detector = DetrObjectDetector(
        model_name=eval_params.get("detector_model_name", "facebook/detr-resnet-50"),
        threshold=eval_params.get("detector_threshold", 0.7),
    )
    max_questions = eval_params.get("max_questions", 4)

    frozen_qa = {sample.sample_id: generate_coco_qa(sample, max_questions=max_questions) for sample in samples}
    write_json(run_dir / "frozen_qa.json", frozen_qa)

    for sample in samples:
        sample.image.save(original_dir / f"{sample.sample_id}.png")

    per_scene_rows: list[dict] = []
    aggregate_buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    comparison_rows: dict[tuple[str, int], dict] = {}
    payload_diagnostics: list[dict] = []

    for method in config.methods:
        codec = build_codec(method, image_size)
        for budget_bytes in config.budgets_bytes:
            method_budget_dir = ensure_dir(recon_dir / method.name / str(budget_bytes))
            method_artifact_dir = ensure_dir(artifact_dir / method.name / str(budget_bytes))
            qa_predictions: list[dict] = []
            for sample in samples:
                encode_start = perf_counter()
                try:
                    artifact = codec.encode(sample, budget_bytes)
                except Exception as exc:
                    per_scene_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "method_name": method.name,
                            "budget_bytes": budget_bytes,
                            "stored_bytes": 0,
                            "budget_utilization": 0.0,
                            "status": f"encode_failed: {exc}",
                        }
                    )
                    continue
                encode_seconds = perf_counter() - encode_start

                decode_start = perf_counter()
                reconstruction = codec.decode(artifact)
                decode_seconds = perf_counter() - decode_start

                recon_path = method_budget_dir / f"{sample.sample_id}.png"
                artifact_path = method_artifact_dir / f"{sample.sample_id}.bin"
                reconstruction.save(recon_path)
                artifact_path.write_bytes(artifact.payload)
                if "text" in artifact.aux:
                    (method_artifact_dir / f"{sample.sample_id}.txt").write_text(artifact.aux["text"])

                qa_score, qa_answers = answer_coco_qa(reconstruction, frozen_qa[sample.sample_id], detector)
                qa_predictions.append({"sample_id": sample.sample_id, "answers": qa_answers})
                metrics = metric_evaluator.evaluate(sample.image, reconstruction)

                row = {
                    "sample_id": sample.sample_id,
                    "method_name": method.name,
                    "budget_bytes": budget_bytes,
                    "stored_bytes": artifact.stored_bytes,
                    "budget_utilization": artifact.stored_bytes / budget_bytes,
                    "scene_qa_accuracy": qa_score,
                    "semantic_similarity": metrics["semantic_similarity"],
                    "dino_similarity": metrics["dino_similarity"],
                    "psnr": metrics["psnr"],
                    "ssim_like": metrics["ssim_like"],
                    "encode_seconds": encode_seconds,
                    "decode_seconds": decode_seconds,
                    "status": "ok",
                }
                if "text" in artifact.aux:
                    row["stored_text"] = artifact.aux["text"]
                    row["payload_sha1"] = artifact.aux.get("text_sha1", hashlib.sha1(artifact.payload).hexdigest())
                    row["segment_count"] = artifact.aux.get("segment_count", 0)
                    payload_diagnostics.append(
                        {
                            "sample_id": sample.sample_id,
                            "method_name": method.name,
                            "budget_bytes": budget_bytes,
                            "stored_bytes": artifact.stored_bytes,
                            "budget_utilization": artifact.stored_bytes / budget_bytes,
                            "payload_sha1": row["payload_sha1"],
                            "stored_text": artifact.aux["text"],
                        }
                    )
                per_scene_rows.append(row)
                aggregate_buckets[(method.name, budget_bytes)].append(row)

                key = (sample.sample_id, budget_bytes)
                if key not in comparison_rows:
                    comparison_rows[key] = {
                        "sample_id": sample.sample_id,
                        "original": {
                            "image_path": f"originals/{sample.sample_id}.png",
                            "stored_bytes": 0,
                            "semantic_similarity": 1.0,
                            "dino_similarity": 1.0,
                            "scene_qa_accuracy": 1.0,
                        },
                    }
                comparison_rows[key][method.name] = {
                    "image_path": f"reconstructions/{method.name}/{budget_bytes}/{sample.sample_id}.png",
                    "stored_bytes": artifact.stored_bytes,
                    "semantic_similarity": metrics["semantic_similarity"],
                    "dino_similarity": metrics["dino_similarity"],
                    "scene_qa_accuracy": qa_score,
                }
            write_json(qa_dir / f"{method.name}_{budget_bytes}.json", qa_predictions)

    aggregate_rows: list[dict] = []
    for (method_name, budget_bytes), rows in sorted(aggregate_buckets.items()):
        count = len(rows)
        if count == 0:
            continue
        aggregate_rows.append(
            {
                "method_name": method_name,
                "budget_bytes": budget_bytes,
                "stored_bytes_mean": sum(row["stored_bytes"] for row in rows) / count,
                "budget_utilization_mean": sum(row["budget_utilization"] for row in rows) / count,
                "scene_qa_accuracy_mean": sum(row["scene_qa_accuracy"] for row in rows) / count,
                "semantic_similarity_mean": sum(row["semantic_similarity"] for row in rows) / count,
                "dino_similarity_mean": sum(row["dino_similarity"] for row in rows) / count,
                "psnr_mean": sum(row["psnr"] for row in rows) / count,
                "ssim_like_mean": sum(row["ssim_like"] for row in rows) / count,
                "encode_seconds_mean": sum(row["encode_seconds"] for row in rows) / count,
                "decode_seconds_mean": sum(row["decode_seconds"] for row in rows) / count,
            }
        )

    comparison_list = [
        row for _, row in sorted(comparison_rows.items()) if "text_only_real" in row and "visual_latent_real" in row
    ]

    identical_payloads: list[dict] = []
    rows_by_sample_method: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_scene_rows:
        if row.get("status") == "ok" and row.get("payload_sha1"):
            rows_by_sample_method[(row["sample_id"], row["method_name"])].append(row)
    for (sample_id, method_name), rows in sorted(rows_by_sample_method.items()):
        rows = sorted(rows, key=lambda item: int(item["budget_bytes"]))
        for prev, curr in zip(rows, rows[1:]):
            if prev.get("payload_sha1") == curr.get("payload_sha1"):
                identical_payloads.append(
                    {
                        "sample_id": sample_id,
                        "method_name": method_name,
                        "budget_a": prev["budget_bytes"],
                        "budget_b": curr["budget_bytes"],
                        "stored_bytes": curr["stored_bytes"],
                        "payload_sha1": curr["payload_sha1"],
                    }
                )

    example_text_payloads = payload_diagnostics[: min(8, len(payload_diagnostics))]

    write_csv(run_dir / "per_scene_metrics.csv", per_scene_rows)
    write_csv(run_dir / "aggregate_metrics.csv", aggregate_rows)
    plot_pareto(aggregate_rows, run_dir / "pareto_curves.png")
    write_comparison_html(comparison_list, run_dir / "comparison.html")
    write_json(diagnostics_dir / "text_payload_diagnostics.json", payload_diagnostics)
    write_json(
        run_dir / "run_summary.json",
        {
            "identical_payloads_across_budgets": identical_payloads,
            "example_text_payloads": example_text_payloads,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_file(args.config)
    run_dir = ensure_dir(Path(config.output_dir) / config.run_name)

    if config.dataset.type == "synthetic_shapes":
        run_synthetic(config, run_dir)
        return

    if config.dataset.type == "coco_subset":
        run_natural(config, run_dir)
        return

    raise ValueError(f"Unsupported dataset type: {config.dataset.type}")


if __name__ == "__main__":
    main()
