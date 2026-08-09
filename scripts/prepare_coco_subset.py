from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve
import zipfile


ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--min-annotations", type=int, default=3)
    parser.add_argument("--min-categories", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ann_dir = output_dir / "annotations"
    img_dir = output_dir / "val2017"
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    zip_path = ann_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        urlretrieve(ANNOTATIONS_ZIP_URL, zip_path)

    extracted_json = ann_dir / "instances_val2017.json"
    if not extracted_json.exists():
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("annotations/instances_val2017.json") as src:
                extracted_json.write_bytes(src.read())

    data = json.loads(extracted_json.read_text())
    categories = {item["id"]: item["name"] for item in data["categories"]}
    images = {item["id"]: item for item in data["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 0:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    selected_image_ids: list[int] = []
    for image_id in sorted(images):
        anns = anns_by_image.get(image_id, [])
        if len(anns) < args.min_annotations:
            continue
        category_count = len({ann["category_id"] for ann in anns})
        if category_count < args.min_categories:
            continue
        selected_image_ids.append(image_id)
        if len(selected_image_ids) >= args.num_samples:
            break

    subset = {
        "images": [images[image_id] for image_id in selected_image_ids],
        "annotations": [ann for image_id in selected_image_ids for ann in anns_by_image.get(image_id, [])],
        "categories": data["categories"],
    }
    subset_path = ann_dir / "instances_val2017_subset.json"
    subset_path.write_text(json.dumps(subset))

    for image_id in selected_image_ids:
        file_name = images[image_id]["file_name"]
        target = img_dir / file_name
        if not target.exists():
            urlretrieve(f"{COCO_IMAGE_BASE_URL}/{file_name}", target)


if __name__ == "__main__":
    main()
