"""Convert COCO polygon segmentation annotations to YOLO Segment format.

Example:
    python tools/dataset/convert_coco_to_yolo_segment.py \
        --coco-json data/datasets/triangle-metal_detect_v1/annotations.json \
        --images-dir data/datasets/triangle-metal_detect_v1 \
        --out-dir data/datasets/triangle-metal_seg_v1
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COCO polygon segmentation dataset to YOLO Segment dataset."
    )
    parser.add_argument(
        "--coco-json",
        type=Path,
        required=True,
        help="Path to COCO annotations.json.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing COCO image files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output YOLO Segment dataset directory.",
    )
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy images into out-dir/images. Default: true.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing out-dir before converting.",
    )
    return parser.parse_args()


def normalize_polygon(segmentation: list[float], width: int, height: int) -> list[str]:
    if len(segmentation) < 6 or len(segmentation) % 2 != 0:
        return []

    values: list[str] = []
    for index in range(0, len(segmentation), 2):
        x = min(max(float(segmentation[index]) / width, 0.0), 1.0)
        y = min(max(float(segmentation[index + 1]) / height, 0.0), 1.0)
        values.append(f"{x:.6f}")
        values.append(f"{y:.6f}")
    return values


def find_image(images_dir: Path, file_name: str) -> Path:
    image_path = images_dir / file_name
    if image_path.is_file():
        return image_path

    candidates = [
        path
        for path in images_dir.rglob(Path(file_name).name)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Image referenced by COCO was not found: {file_name}")
    raise ValueError(f"Multiple images match {file_name}: {candidates}")


def write_data_yaml(out_dir: Path, names: list[str]) -> None:
    lines = [
        f"path: {out_dir.as_posix()}",
        "train: images",
        "val: images",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    coco_json = args.coco_json.resolve()
    images_dir = args.images_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not coco_json.is_file():
        raise SystemExit(f"[ERROR] Missing COCO JSON: {coco_json}")
    if not images_dir.is_dir():
        raise SystemExit(f"[ERROR] Missing images directory: {images_dir}")
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"[ERROR] Output already exists, use --overwrite: {out_dir}")
        shutil.rmtree(out_dir)

    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data.get("images", [])}
    categories = sorted(data.get("categories", []), key=lambda item: item["id"])
    category_id_to_index = {category["id"]: index for index, category in enumerate(categories)}
    names = [category["name"] for category in categories]

    image_lines: dict[int, list[str]] = defaultdict(list)
    skipped_annotations = 0

    for annotation in data.get("annotations", []):
        image = images.get(annotation.get("image_id"))
        if image is None:
            skipped_annotations += 1
            continue

        class_index = category_id_to_index.get(annotation.get("category_id"))
        if class_index is None:
            skipped_annotations += 1
            continue

        segmentations = annotation.get("segmentation") or []
        if isinstance(segmentations, dict):
            skipped_annotations += 1
            continue

        for segmentation in segmentations:
            values = normalize_polygon(segmentation, int(image["width"]), int(image["height"]))
            if not values:
                skipped_annotations += 1
                continue
            image_lines[image["id"]].append(" ".join([str(class_index), *values]))

    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    written_labels = 0
    empty_labels = 0
    for image in sorted(images.values(), key=lambda item: item["file_name"]):
        source_image = find_image(images_dir, image["file_name"])
        label_path = labels_out / f"{Path(image['file_name']).stem}.txt"
        lines = image_lines.get(image["id"], [])
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written_labels += 1
        if not lines:
            empty_labels += 1

        if args.copy_images:
            shutil.copy2(source_image, images_out / source_image.name)
            copied_images += 1

    write_data_yaml(out_dir, names)

    print("[OK] COCO converted to YOLO Segment")
    print(f"[OK] COCO JSON: {coco_json}")
    print(f"[OK] Output: {out_dir}")
    print(f"[OK] Classes: {names}")
    print(f"[OK] Images copied: {copied_images}")
    print(f"[OK] Label files: {written_labels}")
    print(f"[OK] Empty label files: {empty_labels}")
    print(f"[OK] Skipped annotations: {skipped_annotations}")
    print(f"[OK] data.yaml: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
