from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "camera_detect.yaml"


DEFAULTS: dict[str, Any] = {
    "camera": {
        "source": "orbbec",
        "stream": {"width": 1280, "height": 720, "fps": 30.0},
        "opencv": {
            "index": 0,
            "url": None,
            "backend": "auto",
            "read_retries": 30,
            "controls": {
                "auto_exposure": None,
                "exposure": None,
                "gain": None,
            },
        },
        "orbbec": {
            "sdk_dir": "D:/OrbbecSDK_v2",
            "serial": None,
            "format": "RGB",
            "timeout_ms": 1000,
            "depth": {"enabled": True, "align": "sw"},
            "controls": {
                "color_auto_exposure": None,
                "color_exposure": None,
                "color_gain": None,
                "color_ae_max_exposure": None,
                "color_ae_max_gain": None,
                "depth_auto_exposure": None,
                "depth_exposure": None,
                "depth_gain": None,
                "ir_auto_exposure": None,
                "ir_exposure": None,
                "ir_gain": None,
            },
        },
    },
    "model": {
        "path": "runs/detect/data/models/triangle-metal.pt",
        "task": "auto",
        "class_names": None,
    },
    "inference": {
        "imgsz": 960,
        "conf": 0.6,
        "iou": 0.45,
        "device": "0",
        "half": True,
        "infer_stride": 1,
        "max_det": 100,
    },
    "visualize": {
        "draw_masks": True,
        "draw_boxes": True,
        "draw_boxes_when_no_mask": True,
        "draw_labels": True,
        "draw_centers": True,
        "center_mode": "centroid",
        "line_width": 2,
        "mask_alpha": 0.35,
    },
    "display": {
        "backend": "auto",
        "enabled": True,
        "window_title": "YOLO Camera Inference",
        "window_width": "auto",
        "window_height": "auto",
        "resize_to_window": True,
    },
    "output": {
        "save_video": None,
        "snapshot_dir": "data/reports/camera/camera_snapshots",
    },
    "debug": {
        "preview_only": False,
        "print_frame_stats": False,
        "print_detections": False,
    },
    "postprocess": {
        "enabled": False,
        "module": None,
        "function": "process",
        "params": {},
    },
}


@dataclass
class CliArgs:
    config: Path
    dry_run: bool
    list_cameras: bool
    preview_only: bool
    no_window: bool
    overrides: dict[tuple[str, ...], Any]


@dataclass
class RuntimeConfig:
    project_root: Path
    config_path: Path
    profile_name: str
    description: str
    camera: dict[str, Any]
    model: dict[str, Any]
    inference: dict[str, Any]
    visualize: dict[str, Any]
    display: dict[str, Any]
    output: dict[str, Any]
    debug: dict[str, Any]
    postprocess: dict[str, Any]

    def resolve_path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def int_or_auto(value: Any) -> int | str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"auto", "none", "null"}:
        return "auto"
    return int(value)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def move_key(source: dict[str, Any], source_key: str, target: dict[str, Any], target_key: str | None = None) -> None:
    if source_key in source:
        target[target_key or source_key] = source.pop(source_key)


def normalize_common_block(block: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(block)
    camera = normalized.get("camera")
    if isinstance(camera, dict):
        stream = camera.setdefault("stream", {})
        if isinstance(stream, dict):
            for key in ("width", "height", "fps"):
                move_key(camera, key, stream, key)
    return normalized


def normalize_profile_block(block: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(block)

    model_value = normalized.get("model")
    if isinstance(model_value, (str, Path)):
        normalized["model"] = {"path": str(model_value)}
    elif not isinstance(model_value, dict):
        normalized.setdefault("model", {})

    model = normalized["model"]
    if isinstance(model, dict):
        move_key(normalized, "model_path", model, "path")
        move_key(normalized, "weights", model, "path")
        move_key(normalized, "task", model, "task")
        move_key(normalized, "classes", model, "class_names")
        move_key(normalized, "class_names", model, "class_names")

    inference = normalized.setdefault("inference", {})
    if isinstance(inference, dict):
        for key in ("imgsz", "conf", "iou", "device", "half", "infer_stride", "max_det"):
            move_key(normalized, key, inference, key)

    visualize = normalized.setdefault("visualize", {})
    if isinstance(visualize, dict):
        draw_mode = normalized.pop("draw", None)
        if draw_mode is not None:
            apply_draw_mode(visualize, str(draw_mode))
        for key in (
            "draw_masks",
            "draw_boxes",
            "draw_boxes_when_no_mask",
            "draw_labels",
            "draw_centers",
            "center_mode",
            "line_width",
            "mask_alpha",
        ):
            move_key(normalized, key, visualize, key)

    return normalized


def apply_draw_mode(visualize: dict[str, Any], mode: str) -> None:
    normalized = mode.strip().lower()
    if normalized in {"segment", "seg", "mask", "masks"}:
        visualize.update(
            {
                "draw_masks": True,
                "draw_boxes": False,
                "draw_boxes_when_no_mask": True,
                "draw_labels": False,
                "draw_centers": False,
            }
        )
    elif normalized in {"detect", "detection", "box", "boxes"}:
        visualize.update(
            {
                "draw_masks": False,
                "draw_boxes": True,
                "draw_boxes_when_no_mask": True,
                "draw_labels": True,
                "draw_centers": True,
                "center_mode": "box",
            }
        )
    elif normalized in {"full", "all"}:
        visualize.update(
            {
                "draw_masks": True,
                "draw_boxes": True,
                "draw_boxes_when_no_mask": True,
                "draw_labels": True,
                "draw_centers": True,
            }
        )
    else:
        raise SystemExit(f"[ERROR] Unknown draw mode: {mode}. Use segment, detect, or full.")


def parse_cli_args(argv: list[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Run generic YOLO inference from an OpenCV or Orbbec camera."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--no-window", action="store_true")

    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--camera-source", choices=["opencv", "orbbec"], default=None)
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--camera-url", type=str, default=None)
    parser.add_argument("--backend", choices=["auto", "any", "dshow", "msmf", "obsensor"], default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--half", type=str_to_bool, default=None)
    parser.add_argument("--infer-stride", type=int, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--save-video", type=Path, default=None)
    parser.add_argument("--print-detections", action="store_true")

    args = parser.parse_args(argv)
    overrides: dict[tuple[str, ...], Any] = {}
    mapping = {
        "model": ("model", "path"),
        "camera_source": ("camera", "source"),
        "camera_index": ("camera", "opencv", "index"),
        "camera_url": ("camera", "opencv", "url"),
        "backend": ("camera", "opencv", "backend"),
        "width": ("camera", "stream", "width"),
        "height": ("camera", "stream", "height"),
        "fps": ("camera", "stream", "fps"),
        "imgsz": ("inference", "imgsz"),
        "conf": ("inference", "conf"),
        "iou": ("inference", "iou"),
        "device": ("inference", "device"),
        "half": ("inference", "half"),
        "infer_stride": ("inference", "infer_stride"),
        "max_det": ("inference", "max_det"),
        "save_video": ("output", "save_video"),
    }
    for attr, path in mapping.items():
        value = getattr(args, attr)
        if value is not None:
            overrides[path] = value

    if args.preview_only:
        overrides[("debug", "preview_only")] = True
    if args.no_window:
        overrides[("display", "enabled")] = False
    if args.print_detections:
        overrides[("debug", "print_detections")] = True

    return CliArgs(
        config=args.config,
        dry_run=args.dry_run,
        list_cameras=args.list_cameras,
        preview_only=args.preview_only,
        no_window=args.no_window,
        overrides=overrides,
    )


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"[ERROR] Config file not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] Missing dependency: PyYAML. Activate the project environment "
            "or install requirements.txt first."
        ) from exc

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"[ERROR] Config file must contain a YAML mapping: {path}")
    return data


def apply_override(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = data
    for key in path[:-1]:
        next_value = cursor.setdefault(key, {})
        if not isinstance(next_value, dict):
            raise SystemExit(f"[ERROR] Cannot override {'.'.join(path)} because {key} is not a mapping")
        cursor = next_value
    cursor[path[-1]] = str(value) if isinstance(value, Path) else value


def normalize_config_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_runtime_config(cli: CliArgs) -> RuntimeConfig:
    config_path = normalize_config_path(cli.config)
    raw = load_yaml_config(config_path)

    profiles = raw.get("profiles")
    if isinstance(profiles, dict) and profiles:
        profile_name = raw.get("active_profile")
        if not profile_name:
            raise SystemExit("[ERROR] Legacy profile config must define active_profile.")
        if profile_name not in profiles:
            available = ", ".join(sorted(profiles))
            raise SystemExit(f"[ERROR] Unknown active_profile: {profile_name}. Available profiles: {available}")

        common = raw.get("common", {})
        if not isinstance(common, dict):
            raise SystemExit("[ERROR] common must be a YAML mapping.")
        profile = profiles[profile_name]
        if not isinstance(profile, dict):
            raise SystemExit(f"[ERROR] Profile {profile_name!r} must be a YAML mapping.")
        merged = deep_merge(DEFAULTS, normalize_common_block(common))
        merged = deep_merge(merged, normalize_profile_block(profile))
        description = str(profile.get("description", ""))
        runtime_name = str(profile_name)
    else:
        merged = deep_merge(DEFAULTS, normalize_profile_block(normalize_common_block(raw)))
        description = str(raw.get("description", ""))
        runtime_name = ""

    for path, value in cli.overrides.items():
        apply_override(merged, path, value)

    return RuntimeConfig(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        profile_name=runtime_name,
        description=description,
        camera=merged["camera"],
        model=merged["model"],
        inference=merged["inference"],
        visualize=merged["visualize"],
        display=merged["display"],
        output=merged["output"],
        debug=merged["debug"],
        postprocess=merged["postprocess"],
    )
