from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    DEFAULTS,
    PROJECT_ROOT,
    apply_draw_mode,
    deep_merge,
    load_yaml_config,
    normalize_common_block,
    normalize_config_path,
    normalize_profile_block,
)


DEFAULT_MULTI_CONFIG = PROJECT_ROOT / "configs" / "camera_pipelines.yaml"

DEFAULT_CAMERA_ALIASES: dict[str, Any] = {
    "orbbec_335l": {
        "source": "orbbec",
        "orbbec": {"match_name": "Gemini 335L"},
    },
    "orbbec_305": {
        "source": "orbbec",
        "orbbec": {"match_name": "Gemini 305"},
    },
}


@dataclass
class MultiCliArgs:
    config: Path
    dry_run: bool
    list_cameras: bool
    preview_only: bool
    no_window: bool
    pipelines: list[str]


@dataclass
class ModelRuntimeConfig:
    name: str
    model: dict[str, Any]
    inference: dict[str, Any]
    visualize: dict[str, Any]


@dataclass
class PipelineRuntimeConfig:
    name: str
    camera_name: str
    camera: dict[str, Any]
    models: list[ModelRuntimeConfig]
    visualize: dict[str, Any]
    display: dict[str, Any]
    output: dict[str, Any]
    debug: dict[str, Any]
    postprocess: dict[str, Any]


@dataclass
class MultiRuntimeConfig:
    project_root: Path
    config_path: Path
    pipelines: list[PipelineRuntimeConfig]

    def resolve_path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()


def parse_multi_cli_args(argv: list[str] | None = None) -> MultiCliArgs:
    parser = argparse.ArgumentParser(
        description="Run serial YOLO model pipelines across named cameras."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_MULTI_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--pipeline", action="append", default=[], help="Run only the named pipeline. Can be repeated.")
    args = parser.parse_args(argv)

    pipelines = [str(name) for name in args.pipeline]
    return MultiCliArgs(
        config=args.config,
        dry_run=args.dry_run,
        list_cameras=args.list_cameras,
        preview_only=args.preview_only,
        no_window=args.no_window,
        pipelines=pipelines,
    )


def normalize_camera_config(camera_config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_common_block({"camera": copy.deepcopy(camera_config)})
    return normalized.get("camera", {})


def resolve_camera_alias(camera_value: Any, aliases: dict[str, Any], base_camera: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(camera_value, str):
        camera_name = camera_value
        if camera_name not in aliases:
            available = ", ".join(sorted(aliases))
            raise SystemExit(f"[ERROR] Unknown camera alias {camera_name!r}. Available: {available}")
        camera_config = aliases[camera_name]
    elif isinstance(camera_value, dict):
        camera_name = str(camera_value.get("name") or camera_value.get("alias") or "custom_camera")
        alias_name = camera_value.get("alias")
        if alias_name:
            if alias_name not in aliases:
                available = ", ".join(sorted(aliases))
                raise SystemExit(f"[ERROR] Unknown camera alias {alias_name!r}. Available: {available}")
            camera_config = deep_merge(aliases[str(alias_name)], {key: value for key, value in camera_value.items() if key != "alias"})
        else:
            camera_config = camera_value
    else:
        raise SystemExit("[ERROR] pipeline.camera must be a camera alias string or mapping.")

    return camera_name, deep_merge(base_camera, normalize_camera_config(camera_config))


def normalize_visualize_config(source: dict[str, Any], base_visualize: dict[str, Any]) -> dict[str, Any]:
    visual = copy.deepcopy(source.get("visualize", {}) or {})
    draw_mode = source.get("draw")
    if draw_mode is not None:
        apply_draw_mode(visual, str(draw_mode))
    for key in (
        "draw_masks",
        "draw_boxes",
        "draw_boxes_when_no_mask",
        "draw_labels",
        "draw_centers",
        "draw_center_labels",
        "center_mode",
        "line_width",
        "mask_alpha",
    ):
        if key in source:
            visual[key] = copy.deepcopy(source[key])
    return deep_merge(base_visualize, visual)


def normalize_model_item(raw_model: dict[str, Any], base_model: dict[str, Any], base_inference: dict[str, Any]) -> ModelRuntimeConfig:
    if not isinstance(raw_model, dict):
        raise SystemExit("[ERROR] models entries must be YAML mappings.")
    if not bool(raw_model.get("enabled", True)):
        raise SystemExit("[INTERNAL] Disabled model should have been filtered before normalization.")

    block = copy.deepcopy(raw_model)
    model_name = str(block.pop("name", block.get("path", block.get("model", "model"))))
    block.pop("enabled", None)
    block.pop("postprocess", None)
    block.pop("display", None)
    block.pop("output", None)
    if "path" in block and "model" not in block:
        block["model"] = {"path": block.pop("path")}
    if "run_every_n_frames" in block and "infer_stride" not in block:
        block["infer_stride"] = block.pop("run_every_n_frames")

    normalized = normalize_profile_block(block)
    model_config = deep_merge(base_model, normalized.get("model", {}))
    inference_config = deep_merge(base_inference, normalized.get("inference", {}))
    visualize_config = normalized.get("visualize", {}) or {}
    if not isinstance(visualize_config, dict):
        raise SystemExit(f"[ERROR] model {model_name!r}: visualize must be a mapping.")
    model_task = str(model_config.get("task", "auto") or "auto").strip().lower()
    if model_task in {"seg", "segment", "segmentation"}:
        visualize_config.setdefault("draw_masks", True)
    return ModelRuntimeConfig(
        name=model_name,
        model=model_config,
        inference=inference_config,
        visualize=visualize_config,
    )


def pipeline_postprocess_config(raw_pipeline: dict[str, Any], base_postprocess: dict[str, Any]) -> dict[str, Any]:
    postprocess = raw_pipeline.get("postprocess", {})
    if postprocess in (None, False):
        return {"enabled": False, "module": None, "function": "process", "params": {}}
    if not isinstance(postprocess, dict):
        raise SystemExit(f"[ERROR] pipeline {raw_pipeline.get('name', '?')}: postprocess must be a mapping.")
    return deep_merge(base_postprocess, postprocess)


def load_multi_runtime_config(cli: MultiCliArgs) -> MultiRuntimeConfig:
    config_path = normalize_config_path(cli.config)
    raw = load_yaml_config(config_path)

    common = raw.get("common", {})
    if common and not isinstance(common, dict):
        raise SystemExit("[ERROR] common must be a YAML mapping.")
    base = deep_merge(DEFAULTS, normalize_common_block(common or {}))

    aliases = deep_merge(DEFAULT_CAMERA_ALIASES, raw.get("camera_aliases", {}) or {})
    aliases = {str(name): normalize_camera_config(config) for name, config in aliases.items()}

    raw_pipelines = raw.get("pipelines")
    if not isinstance(raw_pipelines, list) or not raw_pipelines:
        raise SystemExit("[ERROR] Multi-camera config must define a non-empty pipelines list.")

    selected = set(cli.pipelines)
    pipelines: list[PipelineRuntimeConfig] = []
    for raw_pipeline in raw_pipelines:
        if not isinstance(raw_pipeline, dict):
            raise SystemExit("[ERROR] pipelines entries must be YAML mappings.")
        if not bool(raw_pipeline.get("enabled", True)):
            continue
        name = str(raw_pipeline.get("name") or f"pipeline_{len(pipelines) + 1}")
        if selected and name not in selected:
            continue

        camera_name, camera = resolve_camera_alias(raw_pipeline.get("camera"), aliases, base["camera"])
        visualize = normalize_visualize_config(raw_pipeline, base["visualize"])
        display = deep_merge(base["display"], raw_pipeline.get("display", {}) or {})
        display.setdefault("window_title", name)
        if display.get("window_title") == DEFAULTS["display"].get("window_title"):
            display["window_title"] = name
        output = deep_merge(base["output"], raw_pipeline.get("output", {}) or {})
        debug = deep_merge(base["debug"], raw_pipeline.get("debug", {}) or {})
        if cli.preview_only:
            debug["preview_only"] = True
        if cli.no_window:
            display["enabled"] = False

        raw_models = raw_pipeline.get("models", [])
        if not isinstance(raw_models, list) or not raw_models:
            raise SystemExit(f"[ERROR] pipeline {name!r} must define at least one model.")
        models = [
            normalize_model_item(raw_model, base["model"], base["inference"])
            for raw_model in raw_models
            if isinstance(raw_model, dict) and bool(raw_model.get("enabled", True))
        ]
        if not models and not bool(debug.get("preview_only", False)):
            raise SystemExit(f"[ERROR] pipeline {name!r} has no enabled models.")

        postprocess = pipeline_postprocess_config(raw_pipeline, base["postprocess"])
        pipelines.append(
            PipelineRuntimeConfig(
                name=name,
                camera_name=camera_name,
                camera=camera,
                models=models,
                visualize=visualize,
                display=display,
                output=output,
                debug=debug,
                postprocess=postprocess,
            )
        )

    if selected:
        found = {pipeline.name for pipeline in pipelines}
        missing = sorted(selected - found)
        if missing:
            raise SystemExit(f"[ERROR] Requested pipeline(s) not enabled or not found: {', '.join(missing)}")
    if not pipelines:
        raise SystemExit("[ERROR] No enabled pipelines to run.")

    return MultiRuntimeConfig(project_root=PROJECT_ROOT, config_path=config_path, pipelines=pipelines)
