#!/usr/bin/env python3
"""Frozen-artifact evaluator and text-encoder robustness controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

SEED = 20260820
ALL_BUDGETS = (256, 512, 1024, 2048, 4096, 8192)
VISUAL_METHODS = ("visual_latent_memory", "webp", "original_raw_oracle")
VISUAL_BUDGETS = (4096, 8192)


def normalize(answer: str, expected: str) -> str:
    value = re.sub(r"[^a-z0-9 ]+", " ", answer.lower()).strip()
    value = re.sub(r"\s+", " ", value)
    expected = re.sub(r"[^a-z0-9 ]+", " ", expected.lower()).strip()
    words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6"}
    if expected.isdigit():
        match = re.search(r"\b[0-9]+\b", value)
        if match:
            return match.group(0)
        return words.get(value, value)
    if expected in {"yes", "no"}:
        match = re.search(r"\b(yes|no)\b", value)
        return match.group(1) if match else value
    prefixes = ("the answer is ", "answer is ", "it is ", "it s ", "a ", "an ", "the ")
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix):]
    return expected if value == expected or value.endswith(" " + expected) else value


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def write_json_atomic(path: Path, payload: dict) -> None:
    """Persist execution telemetry without exposing a partially written file."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_text_control_progress(output: Path, *, completed: int, total: int,
                                phase: str, sample_id: str | None = None,
                                budget: int | None = None) -> None:
    """Heartbeat for execution supervision; it does not affect model inputs or scores."""
    write_json_atomic(output / "progress.json", {
        "schema_version": 1,
        "control": "text_encoder",
        "phase": phase,
        "completed_model_operations": completed,
        "total_model_operations": total,
        "sample_id": sample_id,
        "budget_bytes": budget,
        "updated_at_unix": time.time(),
    })


def validate_inputs(inputs: Path) -> dict:
    provenance = json.loads((inputs / "provenance.json").read_text())
    if provenance["canonical_manifest_sha256"] != "2cd66fcc014c97c51cb07e58407f9aa43e8f124bb59224146ed8b745a407f12e":
        raise RuntimeError("control inputs are not tied to the frozen canonical manifest")
    for record in provenance["files"]:
        path = inputs / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"missing or size-mismatched staged input: {record['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"digest mismatch for staged input: {record['path']}")
    subset = json.loads((inputs / "subset_manifest.json").read_text())
    expected_groups = {"existence": 108, "count": 108, "attribute": 72, "spatial": 108, "state": 72, "relational_reasoning": 108}
    if subset["n_images"] != 144 or subset["n_questions"] != 576 or subset["group_question_counts"] != expected_groups:
        raise RuntimeError("frozen encoder-control subset quota mismatch")
    return {"benchmark": provenance.get("benchmark", "formal_n288"), "verified_files": len(provenance["files"]),
            "subset_images": subset["n_images"], "subset_questions": subset["n_questions"]}


def bootstrap(values: list[float], seed: str) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    draws = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(10_000))
    return draws[249], draws[9749]


def summarize(rows: list[dict], output: Path, *, reference: str | None = None) -> None:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method_name"], row["budget_bytes"])].append(row)
    aggregate = []
    for (method, budget), group in sorted(grouped.items()):
        values = [row["scene_qa_accuracy"] for row in group]
        lo, hi = bootstrap(values, f"{SEED}:{method}:{budget}:aggregate")
        aggregate.append({"method_name": method, "budget_bytes": budget, "n_images": len(group),
                          "stored_bytes_mean": sum(row["stored_bytes"] for row in group) / len(group),
                          "scene_qa_accuracy_mean": sum(values) / len(values),
                          "scene_qa_accuracy_ci95_low": lo, "scene_qa_accuracy_ci95_high": hi})
    write_rows(output / "aggregate_metrics.csv", aggregate)
    if reference is None:
        return
    paired = []
    for budget in sorted({budget for _, budget in grouped}):
        by_method = {method: {row["sample_id"]: row for row in group}
                     for (method, arm_budget), group in grouped.items() if arm_budget == budget}
        if reference not in by_method:
            continue
        for method, current in sorted(by_method.items()):
            if method == reference:
                continue
            ids = sorted(set(current) & set(by_method[reference]))
            differences = [current[sample_id]["scene_qa_accuracy"] - by_method[reference][sample_id]["scene_qa_accuracy"] for sample_id in ids]
            lo, hi = bootstrap(differences, f"{SEED}:{method}:{reference}:{budget}:paired")
            paired.append({"method_name": method, "reference_method": reference, "budget_bytes": budget,
                           "n_paired_images": len(ids), "difference_vs_reference": sum(differences) / len(differences),
                           "paired_image_bootstrap_ci95_low": lo, "paired_image_bootstrap_ci95_high": hi})
    write_rows(output / "pairwise.csv", paired)


def answer_questions(qa: Qwen25VlMemoryQa, sample_id: str, questions: list[dict], *, text: str | None = None,
                     image: Image.Image | None = None, on_answer: Callable[[], None] | None = None) -> list[dict]:
    answers = []
    for question in questions:
        raw = qa.answer(question["question"], text=text, image=image)
        if on_answer is not None:
            on_answer()
        prediction = normalize(raw, question["answer"])
        answers.append({**question, "prediction_raw": raw, "prediction": prediction, "correct": prediction == question["answer"]})
    return answers


def visual_evaluator_control(inputs: Path, output: Path, qa: Qwen25VlMemoryQa) -> None:
    from PIL import Image

    frozen_qa = json.loads((inputs / "frozen_qa.json").read_text())
    with (inputs / "per_scene_metrics.csv").open(newline="") as handle:
        canonical_rates = {(row["sample_id"], row["method_name"], int(row["budget_bytes"])): int(row["stored_bytes"])
                           for row in csv.DictReader(handle) if row["status"] == "ok"}
    rows, predictions = [], []
    for method in VISUAL_METHODS:
        for budget in VISUAL_BUDGETS:
            image_dir = inputs / "reconstructions" / method / str(budget)
            for image_path in sorted(image_dir.glob("*.png")):
                sample_id = image_path.stem
                image = Image.open(image_path).convert("RGB")
                answers = answer_questions(qa, sample_id, frozen_qa[sample_id], image=image)
                rows.append({"control": "second_vlm_evaluator", "sample_id": sample_id, "method_name": method,
                             "budget_bytes": budget, "stored_bytes": canonical_rates[(sample_id, method, budget)],
                             "source_reconstruction_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                             "scene_qa_accuracy": sum(answer["correct"] for answer in answers) / len(answers)})
                predictions.append({"sample_id": sample_id, "method_name": method, "budget_bytes": budget, "answers": answers})
    write_rows(output / "evaluator_per_scene_metrics.csv", rows)
    (output / "evaluator_qa_predictions.json").write_text(json.dumps(predictions, indent=2))
    evaluator_summary = output / "evaluator_summary"
    evaluator_summary.mkdir(exist_ok=True)
    summarize(rows, evaluator_summary, reference="webp")


def text_encoder_control(inputs: Path, output: Path, qa: Qwen25VlMemoryQa, encoder_model: str) -> None:
    from PIL import Image
    from visual_memory_benchmark.codecs.real_text import RealTextCodec
    from visual_memory_benchmark.types import SceneSample

    subset = json.loads((inputs / "subset_manifest.json").read_text())
    if subset["n_images"] != 144 or subset["n_questions"] != 576:
        raise RuntimeError("robustness text control requires frozen N=144 / Q=576 subset")
    image_by_sample = {f"gqa_{item['image_id']}": item for item in subset["images"]}
    baseline_rows, encoded_rows, predictions = [], [], []
    strong_codec = RealTextCodec("qwen25vl_7b_text", 256, vlm_model_name=encoder_model)
    artifact_root = output / "strong_text_artifacts"
    completed_operations = 0
    total_operations = len(subset["sample_ids"]) * len(ALL_BUDGETS) * 9
    checkpoint_path = output / "text_encoder_progress.jsonl"
    write_text_control_progress(output, completed=completed_operations, total=total_operations, phase="starting")

    def completed_operation(*, phase: str, sample_id: str, budget: int) -> None:
        nonlocal completed_operations
        completed_operations += 1
        write_text_control_progress(output, completed=completed_operations, total=total_operations,
                                    phase=phase, sample_id=sample_id, budget=budget)

    for sample_id in subset["sample_ids"]:
        item = image_by_sample[sample_id]
        image_path = Path(item["image_relpath"])
        if not image_path.is_file():
            raise RuntimeError(f"missing original image for frozen subset: {image_path}")
        sample = SceneSample(sample_id=sample_id, image=Image.open(image_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS),
                             objects=None, metadata={"gqa_image_id": item["image_id"]}, source_path=str(image_path))
        for budget in ALL_BUDGETS:
            baseline_path = inputs / "artifacts" / "text_only_memory" / str(budget) / f"{sample_id}.bin"
            baseline_text = baseline_path.read_text(encoding="utf-8")
            baseline_answers = answer_questions(
                qa, sample_id, subset["qa"][sample_id], text=baseline_text,
                on_answer=lambda: completed_operation(phase="baseline_qa", sample_id=sample_id, budget=budget),
            )
            baseline_rows.append({"control": "frontier_text_encoder", "sample_id": sample_id,
                                  "method_name": "frozen_qwen25vl_3b_text", "budget_bytes": budget,
                                  "stored_bytes": baseline_path.stat().st_size,
                                  "artifact_file_bytes": baseline_path.stat().st_size,
                                  "scene_qa_accuracy": sum(answer["correct"] for answer in baseline_answers) / len(baseline_answers)})
            predictions.append({"sample_id": sample_id, "method_name": "frozen_qwen25vl_3b_text", "budget_bytes": budget, "answers": baseline_answers})
            artifact = strong_codec.encode(sample, budget)
            completed_operation(phase="strong_encode", sample_id=sample_id, budget=budget)
            path = artifact_root / str(budget) / f"{sample_id}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(artifact.payload)
            actual = path.stat().st_size
            if actual != artifact.stored_bytes or actual > budget:
                raise RuntimeError(f"new text artifact byte invariant failed for {sample_id} at {budget}")
            strong_answers = answer_questions(
                qa, sample_id, subset["qa"][sample_id], text=artifact.aux["text"],
                on_answer=lambda: completed_operation(phase="strong_qa", sample_id=sample_id, budget=budget),
            )
            encoded_rows.append({"control": "frontier_text_encoder", "sample_id": sample_id,
                                 "method_name": "qwen25vl_7b_text", "budget_bytes": budget,
                                 "stored_bytes": actual, "artifact_file_bytes": actual,
                                 "scene_qa_accuracy": sum(answer["correct"] for answer in strong_answers) / len(strong_answers),
                                 "text_sha256": hashlib.sha256(artifact.payload).hexdigest()})
            predictions.append({"sample_id": sample_id, "method_name": "qwen25vl_7b_text", "budget_bytes": budget, "answers": strong_answers})
            with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
                checkpoint.write(json.dumps({
                    "sample_id": sample_id,
                    "budget_bytes": budget,
                    "baseline_scene_qa_accuracy": baseline_rows[-1]["scene_qa_accuracy"],
                    "strong_scene_qa_accuracy": encoded_rows[-1]["scene_qa_accuracy"],
                    "strong_text_sha256": encoded_rows[-1]["text_sha256"],
                }) + "\n")
    rows = baseline_rows + encoded_rows
    write_rows(output / "text_encoder_per_scene_metrics.csv", rows)
    (output / "text_encoder_qa_predictions.json").write_text(json.dumps(predictions, indent=2))
    encoder_summary = output / "text_encoder_summary"
    encoder_summary.mkdir(exist_ok=True)
    summarize(rows, encoder_summary, reference="frozen_qwen25vl_3b_text")
    write_text_control_progress(output, completed=completed_operations, total=total_operations, phase="completed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="data/formal_n288_robustness_controls")
    parser.add_argument("--output", default="outputs/robustness_controls")
    parser.add_argument("--control", choices=("evaluator", "text_encoder"), required=True)
    parser.add_argument("--qa-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--encoder-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--validate-inputs", action="store_true")
    args = parser.parse_args()
    inputs, output = Path(args.inputs), Path(args.output)
    validated = validate_inputs(inputs)
    if args.validate_inputs:
        print(json.dumps(validated, indent=2))
        return
    from visual_memory_benchmark.models.hf_adapters import Qwen25VlMemoryQa

    provenance = json.loads((inputs / "provenance.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "benchmark": provenance.get("benchmark", "formal_n288"), "canonical_manifest_sha256": provenance["canonical_manifest_sha256"],
        "control": args.control, "qa_model": args.qa_model,
        "bootstrap": {"resamples": 10000, "unit": "image", "seed": SEED},
    }
    if args.control == "evaluator":
        protocol["restriction"] = "Only the evaluator model changes. Frozen visual reconstructions, questions, normalization, scoring, and byte records are read only; no artifact is encoded, decoded, resized, or regenerated."
    else:
        protocol["text_encoder_model"] = args.encoder_model
        protocol["restriction"] = "Existing visual/text baseline artifacts are read only; only qwen25vl_7b_text is newly encoded on the frozen N=144 subset."
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    qa = Qwen25VlMemoryQa(model_name=args.qa_model)
    if args.control == "evaluator":
        visual_evaluator_control(inputs, output, qa)
    else:
        text_encoder_control(inputs, output, qa, args.encoder_model)


if __name__ == "__main__":
    main()
