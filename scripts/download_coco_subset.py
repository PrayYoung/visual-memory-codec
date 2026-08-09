from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON file with {annotations_url, images:[{url,file_name,id}]}")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "val2017"
    ann_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    if "annotations_url" in manifest and "annotations_file" in manifest:
        target = ann_dir / manifest["annotations_file"]
        if not target.exists():
            urlretrieve(manifest["annotations_url"], target)

    for item in manifest["images"]:
        target = images_dir / item["file_name"]
        if not target.exists():
            urlretrieve(item["url"], target)


if __name__ == "__main__":
    main()
