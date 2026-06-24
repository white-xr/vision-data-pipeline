"""Export manually selected FiftyOne samples for annotation.

Example:
    python tools/export_annotation_candidates.py \
        --dataset-name hole_review_v1 \
        --out-dir data/annotation/hole_detect_v1 \
        --include-tags to_annotate hard
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Iterable

import fiftyone as fo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy FiftyOne samples tagged for annotation into an annotation image directory."
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="FiftyOne dataset name created by tools/view_with_fiftyone.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory, for example data/annotation/hole_detect_v1.",
    )
    parser.add_argument(
        "--include-tags",
        nargs="+",
        required=True,
        help="Sample tags to export, for example: to_annotate hard.",
    )
    return parser.parse_args()


def safe_token(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value or "unknown"


def unique_output_path(images_dir: Path, batch_name: str, filename: str) -> Path:
    src_name = Path(filename)
    base = f"{safe_token(batch_name)}_{safe_token(src_name.stem)}"
    suffix = src_name.suffix.lower()
    candidate = images_dir / f"{base}{suffix}"
    counter = 1

    while candidate.exists():
        candidate = images_dir / f"{base}_{counter:03d}{suffix}"
        counter += 1

    return candidate


def sample_has_any_tag(sample_tags: Iterable[str], include_tags: set[str]) -> bool:
    return bool(set(sample_tags) & include_tags)


def get_sample_value(sample: fo.Sample, field_name: str, default: str) -> str:
    try:
        value = sample[field_name]
    except Exception:
        value = None
    return str(value or default)


def export_candidates(dataset_name: str, out_dir: Path, include_tags: list[str]) -> int:
    if not fo.dataset_exists(dataset_name):
        raise SystemExit(f"[ERROR] FiftyOne dataset does not exist: {dataset_name}")

    dataset = fo.load_dataset(dataset_name)
    include_tag_set = set(include_tags)
    out_dir = out_dir.resolve()
    images_dir = out_dir / "images"
    manifest_path = out_dir / "export_manifest.csv"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for sample in dataset:
        if "bad" in sample.tags:
            continue
        if not sample_has_any_tag(sample.tags, include_tag_set):
            continue

        source_path = Path(sample.filepath)
        if not source_path.is_file():
            print(f"[WARN] Source image missing, skipped: {source_path}")
            continue

        batch_name = get_sample_value(sample, "batch_name", source_path.parent.name)
        filename = get_sample_value(sample, "filename", source_path.name)
        output_path = unique_output_path(images_dir, batch_name, filename)

        shutil.copy2(source_path, output_path)
        rows.append(
            {
                "source_path": str(source_path),
                "output_path": str(output_path),
                "batch_name": str(batch_name),
                "filename": str(filename),
                "tags": " ".join(sorted(sample.tags)),
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_path", "output_path", "batch_name", "filename", "tags"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Exported images: {len(rows)}")
    print(f"[OK] Images directory: {images_dir}")
    print(f"[OK] Manifest: {manifest_path}")
    return len(rows)


def main() -> None:
    args = parse_args()
    export_candidates(args.dataset_name, args.out_dir, args.include_tags)


if __name__ == "__main__":
    main()
