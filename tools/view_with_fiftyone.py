"""Load raw images into FiftyOne for manual review.

Example:
    python tools/view_with_fiftyone.py --raw-dir data/raw --dataset-name hole_review_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fiftyone as fo


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_SPLIT_TAG = "unreviewed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load data/raw images into a FiftyOne dataset for manual review."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw image root directory. Each first-level directory is treated as a batch.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="FiftyOne dataset name, for example hole_review_v1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the dataset if it already exists.",
    )
    return parser.parse_args()


def iter_image_paths(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        raise SystemExit(f"[ERROR] Raw directory does not exist: {raw_dir}")

    image_paths = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(image_paths, key=lambda p: p.relative_to(raw_dir).as_posix().lower())


def get_batch_name(raw_dir: Path, image_path: Path) -> str:
    rel_path = image_path.relative_to(raw_dir)
    if len(rel_path.parts) <= 1:
        return raw_dir.name
    return rel_path.parts[0]


def create_dataset(raw_dir: Path, dataset_name: str, overwrite: bool) -> fo.Dataset:
    raw_dir = raw_dir.resolve()

    if fo.dataset_exists(dataset_name):
        if not overwrite:
            print(
                f"[INFO] FiftyOne dataset already exists: {dataset_name}\n"
                "[INFO] Loading existing dataset to preserve manual review tags.\n"
                "[INFO] Use --overwrite only when you intentionally want to recreate it."
            )
            return fo.load_dataset(dataset_name)

        print(f"[INFO] Deleting existing FiftyOne dataset: {dataset_name}")
        fo.delete_dataset(dataset_name)

    image_paths = iter_image_paths(raw_dir)
    if not image_paths:
        raise SystemExit(f"[ERROR] No images found under: {raw_dir}")

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    dataset.info = {
        "raw_dir": str(raw_dir),
        "purpose": "manual image review before annotation",
    }

    samples = []
    for image_path in image_paths:
        sample = fo.Sample(filepath=str(image_path))
        sample["batch_name"] = get_batch_name(raw_dir, image_path)
        sample["filename"] = image_path.name
        sample["split_tag"] = DEFAULT_SPLIT_TAG
        sample.tags.append(DEFAULT_SPLIT_TAG)
        samples.append(sample)

    dataset.add_samples(samples)
    print(f"[OK] Created dataset: {dataset_name}")
    print(f"[OK] Loaded images: {len(samples)}")
    return dataset


def main() -> None:
    args = parse_args()
    dataset = create_dataset(args.raw_dir, args.dataset_name, args.overwrite)

    print("[INFO] Launching FiftyOne App...")
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()

