"""Copy raw images into an AnyLabeling annotation workspace.

Example:
    python tools/dataset/prepare_anylabeling_dataset.py \
        --raw-dir data/raw \
        --out-dir data/annotation/hole_detect_v1
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare images from data/raw for AnyLabeling YOLO detection annotation."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw image root directory. Each first-level directory is treated as a batch.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/annotation/hole_detect_v1"),
        help="Annotation output root. Images are copied into <out-dir>/images.",
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


def batch_name_for(raw_dir: Path, image_path: Path) -> str:
    rel_path = image_path.relative_to(raw_dir)
    if len(rel_path.parts) <= 1:
        return raw_dir.name
    return rel_path.parts[0]


def safe_token(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value or "unknown"


def unique_output_path(images_dir: Path, batch_name: str, filename: str) -> Path:
    source_name = Path(filename)
    base_name = f"{safe_token(batch_name)}_{safe_token(source_name.stem)}"
    suffix = source_name.suffix.lower()
    candidate = images_dir / f"{base_name}{suffix}"
    counter = 1

    while candidate.exists():
        candidate = images_dir / f"{base_name}_{counter:03d}{suffix}"
        counter += 1

    return candidate


def prepare_anylabeling_dataset(raw_dir: Path, out_dir: Path) -> int:
    raw_dir = raw_dir.resolve()
    out_dir = out_dir.resolve()
    images_dir = out_dir / "images"
    manifest_path = out_dir / "prepare_manifest.csv"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for source_path in iter_image_paths(raw_dir):
        batch_name = batch_name_for(raw_dir, source_path)
        filename = source_path.name
        output_path = unique_output_path(images_dir, batch_name, filename)

        shutil.copy2(source_path, output_path)
        rows.append(
            {
                "source_path": str(source_path),
                "output_path": str(output_path),
                "batch_name": batch_name,
                "filename": filename,
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_path", "output_path", "batch_name", "filename"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Copied images: {len(rows)}")
    print(f"[OK] Images directory: {images_dir}")
    print(f"[OK] Manifest: {manifest_path}")
    return len(rows)


def main() -> None:
    args = parse_args()
    prepare_anylabeling_dataset(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
