"""Split annotated YOLO Detect data into train/val directories.

Example:
    python tools/dataset/split_yolo_dataset.py \
        --src-dir data/annotation/hole_detect_v1 \
        --out-dir data/datasets/hole_detect_v1 \
        --train-ratio 0.8 \
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
CLASS_NAMES = ["cover_edge_hole", "base_edge_hole"]


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split annotated images and YOLO labels into train/val datasets."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/annotation/hole_detect_v1"),
        help="Annotation root containing images/ and labels/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/datasets/hole_detect_v1"),
        help="Output YOLO Detect dataset root.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio. Default: 0.8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting. Default: 42.",
    )
    return parser.parse_args()


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for suffix in sorted(IMAGE_SUFFIXES):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate

    matches = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple images match label stem {stem!r}: {matches}")
    return matches[0] if matches else None


def validate_label_file(label_path: Path) -> None:
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_no} is not YOLO Detect format with 5 fields")
        if parts[0] not in {"0", "1"}:
            raise ValueError(f"{label_path}:{line_no} has invalid class id {parts[0]!r}")
        for value in parts[1:]:
            number = float(value)
            if number < 0.0 or number > 1.0:
                raise ValueError(f"{label_path}:{line_no} coordinate out of range [0, 1]: {value}")


def collect_samples(src_dir: Path) -> list[Sample]:
    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"
    if not images_dir.is_dir():
        raise SystemExit(f"[ERROR] Missing images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"[ERROR] Missing labels directory: {labels_dir}")

    samples: list[Sample] = []
    missing_images: list[Path] = []

    label_files = sorted(
        path
        for path in labels_dir.glob("*.txt")
        if path.name not in {"classes.txt"}
    )
    for label_path in label_files:
        validate_label_file(label_path)
        image_path = find_image_for_label(images_dir, label_path.stem)
        if image_path is None:
            missing_images.append(label_path)
            continue
        samples.append(Sample(image_path=image_path, label_path=label_path))

    if missing_images:
        print("[ERROR] Labels without matching images:")
        for path in missing_images[:50]:
            print(f"  {path}")
        raise SystemExit(f"[ERROR] Found {len(missing_images)} labels without matching images.")

    if not samples:
        raise SystemExit("[ERROR] No labeled samples found.")

    return samples


def split_samples(samples: list[Sample], train_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise SystemExit("[ERROR] --train-ratio must be between 0 and 1.")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    train_count = round(len(shuffled) * train_ratio)
    train_count = max(1, min(len(shuffled) - 1, train_count))
    return shuffled[:train_count], shuffled[train_count:]


def prepare_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_split(samples: list[Sample], out_dir: Path, split: str) -> None:
    for sample in samples:
        shutil.copy2(sample.image_path, out_dir / "images" / split / sample.image_path.name)
        label_text = sample.label_path.read_text(encoding="utf-8-sig")
        (out_dir / "labels" / split / sample.label_path.name).write_text(label_text, encoding="utf-8")


def write_data_yaml(out_dir: Path) -> None:
    lines = [
        "path: data/datasets/hole_detect_v1",
        "train: images/train",
        "val: images/val",
        "nc: 2",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    src_dir = args.src_dir.resolve()
    out_dir = args.out_dir.resolve()

    samples = collect_samples(src_dir)
    train_samples, val_samples = split_samples(samples, args.train_ratio, args.seed)
    prepare_output_dir(out_dir)
    copy_split(train_samples, out_dir, "train")
    copy_split(val_samples, out_dir, "val")
    write_data_yaml(out_dir)

    print("[OK] YOLO Detect dataset created")
    print(f"[OK] Source samples: {len(samples)}")
    print(f"[OK] Train samples: {len(train_samples)}")
    print(f"[OK] Val samples: {len(val_samples)}")
    print(f"[OK] Output: {out_dir}")
    print(f"[OK] data.yaml: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
