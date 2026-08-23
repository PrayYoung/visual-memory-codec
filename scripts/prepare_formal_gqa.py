#!/usr/bin/env python3
"""Freeze and stage the formal N=288 GQA/Visual Genome QA manifest.

This is dataset preparation only.  It neither imports nor invokes a memory
encoder, decoder, codec, evaluator, or any benchmark metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

SEED = 20260820
TEMPLATES = (("C", 72, ("spatial", "relational_reasoning", "attribute", "state")),
             ("B", 72, ("existence", "count", "attribute", "state")),
             ("A", 144, ("existence", "count", "spatial", "relational_reasoning")))
URLS = {
    "questions": "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip#val_balanced_questions.json",
    "scene_graphs": "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip#val_sceneGraphs.json",
    "images": "https://cs.stanford.edu/people/rak248/VG_100K[/_2]/{image_id}.jpg",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: str) -> str:
    return " ".join(value.lower().split())


def classify(question: dict) -> str | None:
    detailed = question["types"]["detailed"]
    semantic = question["types"]["semantic"]
    answer = norm(question["answer"])
    steps = question["semantic"]
    if detailed.startswith("exist") and answer in {"yes", "no"}:
        return "existence"
    if detailed.startswith("position") and answer in {"left", "right", "top", "bottom", "yes", "no"}:
        return "spatial"
    # activity questions directly query an action/state.  activityWho is not a
    # state question and is intentionally excluded.
    if detailed == "activity" and answer not in {"yes", "no", "unknown"}:
        return "state"
    if semantic == "attr" and answer not in {"yes", "no", "unknown"}:
        return "attribute"
    # Require composition beyond a one-hop relation lookup.
    if semantic == "rel" and (question["types"]["structural"] in {"logical", "compare"}
                              or sum(step["operation"] == "relate" for step in steps) >= 2):
        return "relational_reasoning"
    return None


def count_item(image_id: str, graph: dict) -> dict | None:
    counts = Counter(norm(obj.get("name", "")) for obj in graph["objects"].values())
    candidates = sorted((label, count) for label, count in counts.items() if label and 2 <= count <= 6)
    if not candidates:
        return None
    label, count = candidates[0]
    return {
        "group": "count", "source": "gqa_scene_graphs_v1.1", "source_question_id": None,
        "question": f"How many {label} objects are visible?", "answer": str(count),
        "answer_type": "integer", "semantic": [{"operation": "count", "argument": label}],
    }


def gqa_item(question_id: str, question: dict, group: str) -> dict:
    return {
        "group": group, "source": "gqa_balanced_val_questions_v1.2", "source_question_id": question_id,
        "question": question["question"], "answer": norm(question["answer"]), "answer_type": "short_answer",
        "types": question["types"], "semantic": question["semantic"],
    }


def manifest(root: Path) -> dict:
    qpath = root / "data/gqa/raw/val_balanced_questions.json"
    gpath = root / "data/gqa/raw/val_sceneGraphs.json"
    questions = json.loads(qpath.read_text())
    graphs = json.loads(gpath.read_text())
    candidates: dict[str, dict[str, list[tuple[str, dict]]]] = defaultdict(lambda: defaultdict(list))
    for qid, question in questions.items():
        image_id = question["imageId"]
        group = classify(question)
        if group is not None and image_id in graphs:
            candidates[image_id][group].append((qid, gqa_item(qid, question, group)))
    for image_id, graph in graphs.items():
        item = count_item(image_id, graph)
        if item is not None:
            candidates[image_id]["count"].append(("scene_graph_count", item))
    for groups in candidates.values():
        for items in groups.values():
            items.sort(key=lambda pair: pair[0])

    rng = random.Random(SEED)
    used: set[str] = set()
    chosen: list[tuple[str, str, tuple[str, ...]]] = []
    for template, target, group_tuple in TEMPLATES:
        pool = [image_id for image_id, groups in candidates.items()
                if image_id not in used and all(groups.get(group) for group in group_tuple)]
        rng.shuffle(pool)
        if len(pool) < target:
            raise RuntimeError(f"Only {len(pool)} candidates for template {template}; need {target}")
        for image_id in pool[:target]:
            used.add(image_id)
            chosen.append((image_id, template, group_tuple))
    if len(chosen) != 288:
        raise RuntimeError(f"Selected {len(chosen)}, expected 288")

    images = []
    for image_id, template, groups in sorted(chosen, key=lambda row: int(row[0])):
        qa = []
        for group in groups:
            options = candidates[image_id][group]
            qa.append(rng.choice(options)[1])
        images.append({"image_id": image_id, "template": template,
                       "image_relpath": f"data/gqa/images/{image_id}.jpg", "qa": qa})
    totals = Counter(item["group"] for image in images for item in image["qa"])
    expected = {"existence": 216, "count": 216, "attribute": 144, "spatial": 216,
                "state": 144, "relational_reasoning": 216}
    if dict(totals) != expected:
        raise RuntimeError(f"Quota mismatch: {dict(totals)} != {expected}")
    return {
        "schema_version": 1, "seed": SEED, "dataset": "GQA balanced val v1.2 / scene graphs v1.1 (Visual Genome images)",
        "provenance": URLS, "source_sha256": {"questions": sha256(qpath), "scene_graphs": sha256(gpath)},
        "n_images": len(images), "n_questions": sum(len(image["qa"]) for image in images),
        "group_question_counts": dict(sorted(totals.items())), "selection_templates": {name: n for name, n, _ in TEMPLATES},
        "images": images,
    }


def download_images(root: Path, data: dict) -> None:
    from PIL import Image
    image_dir = root / "data/gqa/images"; image_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(data["images"], start=1):
        target = root / image["image_relpath"]
        if not target.exists():
            image_id = image["image_id"]
            urls = [f"https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg",
                    f"https://cs.stanford.edu/people/rak248/VG_100K_2/{image_id}.jpg"]
            error = None
            for url in urls:
                try:
                    with urlopen(url, timeout=60) as response:
                        target.write_bytes(response.read())
                    break
                except HTTPError as exc:
                    error = exc
            else:
                raise RuntimeError(f"Could not fetch {image_id}: {error}")
            time.sleep(0.03)
        with Image.open(target) as decoded:
            decoded.verify()
        image["image_sha256"] = sha256(target)
        if index % 25 == 0:
            print(f"staged {index}/{len(data['images'])}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage-images", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    data = manifest(root)
    if args.stage_images:
        download_images(root, data)
    output = root / "data/gqa/formal_n288_manifest.json"
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: data[key] for key in ("n_images", "n_questions", "group_question_counts", "source_sha256")}, indent=2))
    print(f"manifest={output}")
    print(f"manifest_sha256={sha256(output)}")


if __name__ == "__main__":
    main()
