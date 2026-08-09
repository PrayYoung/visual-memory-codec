from __future__ import annotations

from collections import Counter
from typing import Any

from PIL import Image

from visual_memory_benchmark.models.hf_adapters import DetrObjectDetector
from visual_memory_benchmark.types import SceneSample


def generate_coco_qa(sample: SceneSample, max_questions: int = 4) -> list[dict[str, Any]]:
    anns = sample.metadata["annotations"]
    categories = sample.metadata["categories"]
    counts = Counter(categories[ann["category_id"]] for ann in anns)
    ordered_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    qa_pairs: list[dict[str, Any]] = []

    for label, count in ordered_counts[:2]:
        qa_pairs.append(
            {
                "question": f"How many {label} objects are visible?",
                "answer": str(count),
                "type": "count",
                "label": label,
            }
        )

    if ordered_counts:
        label = ordered_counts[0][0]
        qa_pairs.append(
            {
                "question": f"Is there a {label} visible?",
                "answer": "yes",
                "type": "presence",
                "label": label,
            }
        )

    relation_pair = _largest_distinct_pair(anns, categories)
    if relation_pair is not None:
        a, b = relation_pair
        qa_pairs.append(
            {
                "question": f"Is the {a['label']} to the left or right of the {b['label']}?",
                "answer": _relation(a["bbox"], b["bbox"]),
                "type": "relation",
                "left_label": a["label"],
                "right_label": b["label"],
            }
        )

    return qa_pairs[:max_questions]


def answer_coco_qa(image: Image.Image, qa_pairs: list[dict[str, Any]], detector: DetrObjectDetector) -> tuple[float, list[dict[str, Any]]]:
    detections = detector.detect(image)
    answers: list[dict[str, Any]] = []
    correct = 0
    counts = Counter(item["label"] for item in detections)

    for qa in qa_pairs:
        predicted = "unknown"
        if qa["type"] == "count":
            predicted = str(counts.get(qa["label"], 0))
        elif qa["type"] == "presence":
            predicted = "yes" if counts.get(qa["label"], 0) > 0 else "no"
        elif qa["type"] == "relation":
            a = _best_detection(detections, qa["left_label"])
            b = _best_detection(detections, qa["right_label"])
            if a is not None and b is not None:
                predicted = _relation(a["bbox"], b["bbox"])
        is_correct = predicted == qa["answer"]
        correct += int(is_correct)
        answers.append(
            {
                "question": qa["question"],
                "reference_answer": qa["answer"],
                "predicted_answer": predicted,
                "correct": is_correct,
            }
        )

    score = correct / len(qa_pairs) if qa_pairs else 0.0
    return score, answers


def _largest_distinct_pair(anns: list[dict], categories: dict[int, str]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    converted = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        converted.append(
            {
                "bbox": [x, y, x + w, y + h],
                "area": float(ann.get("area", w * h)),
                "label": categories[ann["category_id"]],
            }
        )
    converted.sort(key=lambda item: item["area"], reverse=True)
    for i, a in enumerate(converted):
        for b in converted[i + 1 :]:
            if a["label"] != b["label"]:
                return a, b
    return None


def _best_detection(detections: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    matches = [item for item in detections if item["label"] == label]
    if not matches:
        return None
    return max(matches, key=lambda item: item["score"])


def _relation(box_a: list[float], box_b: list[float]) -> str:
    ax = (box_a[0] + box_a[2]) / 2.0
    bx = (box_b[0] + box_b[2]) / 2.0
    return "left" if ax < bx else "right"
