from __future__ import annotations

import math
from collections import Counter

import numpy as np
from PIL import Image

from visual_memory_benchmark.data.synthetic_shapes import COLORS, SHAPES, relation
from visual_memory_benchmark.types import SceneObject, SceneSample


def psnr(original: Image.Image, reconstructed: Image.Image) -> float:
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(reconstructed.resize(original.size), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_like(original: Image.Image, reconstructed: Image.Image) -> float:
    a = np.asarray(original.convert("L"), dtype=np.float32)
    b = np.asarray(reconstructed.resize(original.size).convert("L"), dtype=np.float32)
    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    sigma_a = float(np.var(a))
    sigma_b = float(np.var(b))
    sigma_ab = float(np.mean((a - mu_a) * (b - mu_b)))
    c1 = 6.5025
    c2 = 58.5225
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return numerator / denominator if denominator else 0.0


def infer_objects_from_image(image: Image.Image) -> list[SceneObject]:
    array = np.asarray(image, dtype=np.uint8)
    h, w, _ = array.shape
    objects: list[SceneObject] = []
    for color_name, rgb in COLORS.items():
        mask = np.all(np.abs(array.astype(np.int16) - np.asarray(rgb, dtype=np.int16)) <= 25, axis=-1)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        left, right = int(xs.min()), int(xs.max())
        top, bottom = int(ys.min()), int(ys.max())
        bbox = (left, top, right, bottom)
        fill_ratio = float(mask[top : bottom + 1, left : right + 1].mean())
        shape = _infer_shape(fill_ratio)
        objects.append(SceneObject(shape=shape, color=color_name, bbox=bbox))
    return objects


def semantic_recall(sample: SceneSample, reconstructed: Image.Image) -> float:
    gt = Counter((obj.color, obj.shape) for obj in sample.objects)
    pred = Counter((obj.color, obj.shape) for obj in infer_objects_from_image(reconstructed))
    total = sum(gt.values())
    if total == 0:
        return 0.0
    matched = 0
    for key, count in gt.items():
        matched += min(count, pred.get(key, 0))
    return matched / total


def scene_fidelity(sample: SceneSample, reconstructed: Image.Image) -> float:
    predicted = infer_objects_from_image(reconstructed)
    if not sample.objects:
        return 0.0
    used: set[int] = set()
    pair_scores: list[float] = []
    for gt in sample.objects:
        best_idx = None
        best_score = 0.0
        for idx, pred in enumerate(predicted):
            if idx in used or pred.color != gt.color or pred.shape != gt.shape:
                continue
            score = _bbox_iou(gt.bbox, pred.bbox)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
            pair_scores.append(best_score)
        else:
            pair_scores.append(0.0)
    return float(sum(pair_scores) / len(sample.objects))


def generate_scene_qa(sample: SceneSample) -> list[dict[str, str]]:
    qa_pairs: list[dict[str, str]] = []
    shape_counts = Counter(obj.shape for obj in sample.objects)
    for shape in SHAPES:
        qa_pairs.append(
            {
                "question": f"How many {shape}s are visible?",
                "answer": str(shape_counts.get(shape, 0)),
                "type": "count",
                "target": shape,
            }
        )
    if len(sample.objects) >= 2:
        a, b = sample.objects[0], sample.objects[1]
        qa_pairs.append(
            {
                "question": f"Is the {a.color} {a.shape} left_of, right_of, above, or below the {b.color} {b.shape}?",
                "answer": relation(a, b),
                "type": "relation",
                "target": "pair0",
            }
        )
    first = sample.objects[0]
    qa_pairs.append(
        {
            "question": f"What color is the first object?",
            "answer": first.color,
            "type": "attribute",
            "target": "first_color",
        }
    )
    return qa_pairs


def answer_scene_qa(reconstructed: Image.Image, sample: SceneSample, qa_pairs: list[dict[str, str]]) -> float:
    predicted = infer_objects_from_image(reconstructed)
    pred_shape_counts = Counter(obj.shape for obj in predicted)
    score = 0
    for item in qa_pairs:
        if item["type"] == "count":
            pred = str(pred_shape_counts.get(item["target"], 0))
            score += int(pred == item["answer"])
        elif item["type"] == "relation":
            if len(predicted) >= 2:
                pred = relation(predicted[0], predicted[1])
                score += int(pred == item["answer"])
        elif item["type"] == "attribute":
            pred = predicted[0].color if predicted else "unknown"
            score += int(pred == item["answer"])
    return score / len(qa_pairs) if qa_pairs else 0.0


def _infer_shape(fill_ratio: float) -> str:
    if fill_ratio > 0.9:
        return "square"
    if fill_ratio > 0.68:
        return "circle"
    return "triangle"


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)
