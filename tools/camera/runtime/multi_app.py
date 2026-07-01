from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from .app import create_writer, print_detections, save_snapshot
from .camera import list_cameras, open_camera
from .config import DEFAULTS
from .display import create_display
from .geometry import prepare_depth_for_lookup
from .inference import YoloRunner, detections_from_result
from .multi_config import (
    ModelRuntimeConfig,
    MultiRuntimeConfig,
    PipelineRuntimeConfig,
    load_multi_runtime_config,
    parse_multi_cli_args,
)
from .postprocess import PostProcessor, load_post_processor
from .visualize import draw_detections, draw_overlays, draw_status_lines


@dataclass
class SerialModelRunner:
    name: str
    runner: YoloRunner
    infer_stride: int
    visualize: dict[str, Any]
    last_detections: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineRuntime:
    config: PipelineRuntimeConfig
    capture: Any
    models: list[SerialModelRunner]
    post_processor: PostProcessor
    display: Any = None
    writer: cv2.VideoWriter | None = None
    frame_id: int = 0
    started_at: float = field(default_factory=time.time)
    warned_depth_unaligned: bool = False
    active: bool = True


def resolve_path(project_root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def make_model_runner(model_config: ModelRuntimeConfig, project_root: Path, preview_only: bool) -> SerialModelRunner:
    runner = YoloRunner(model_config.model, model_config.inference, project_root)
    if not preview_only:
        runner.load()
    infer_stride = max(1, int(model_config.inference.get("infer_stride", 1)))
    return SerialModelRunner(
        name=model_config.name,
        runner=runner,
        infer_stride=infer_stride,
        visualize=model_config.visualize,
    )


def dry_run(config: MultiRuntimeConfig) -> None:
    print("[OK] Multi-camera dry run passed.")
    print(f"[OK] Config: {config.config_path}")
    for pipeline in config.pipelines:
        print(f"[OK] Pipeline: {pipeline.name}, camera={pipeline.camera_name}")
        post_processor = load_post_processor(pipeline.postprocess)
        print(f"[OK]   Postprocess: {post_processor.label}")
        if bool(pipeline.debug.get("preview_only", False)):
            print("[OK]   Preview only: models are not validated.")
            continue
        for model in pipeline.models:
            runner = YoloRunner(model.model, model.inference, config.project_root)
            runner.validate_model_path()
            print(
                "[OK]   Model: "
                f"{model.name}, path={runner.model_path}, "
                f"task={model.model.get('task', 'auto')}, "
                f"imgsz={model.inference.get('imgsz')}, conf={model.inference.get('conf')}, "
                f"infer_stride={model.inference.get('infer_stride')}"
            )


def open_pipeline_runtime(pipeline: PipelineRuntimeConfig, project_root: Path) -> PipelineRuntime:
    preview_only = bool(pipeline.debug.get("preview_only", False))
    capture = open_camera(pipeline.camera)
    model_runners = [
        make_model_runner(model, project_root, preview_only)
        for model in pipeline.models
    ]
    post_processor = load_post_processor(pipeline.postprocess)
    if preview_only:
        print(f"[OK] Pipeline {pipeline.name}: preview only, camera={pipeline.camera_name}")
    else:
        model_names = ", ".join(model.name for model in model_runners)
        print(f"[OK] Pipeline {pipeline.name}: camera={pipeline.camera_name}, models={model_names}")
    return PipelineRuntime(
        config=pipeline,
        capture=capture,
        models=model_runners,
        post_processor=post_processor,
    )


def model_should_infer(model: SerialModelRunner, frame_id: int) -> bool:
    return frame_id == 1 or (frame_id - 1) % model.infer_stride == 0


def run_serial_models(
    runtime: PipelineRuntime,
    frame: Any,
    depth_for_lookup: Any,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for model in runtime.models:
        if model_should_infer(model, runtime.frame_id):
            center_mode = str(model.visualize.get("center_mode", runtime.config.visualize.get("center_mode", "centroid")))
            result = model.runner.predict(frame)
            detections = detections_from_result(
                result,
                frame.shape,
                depth_for_lookup,
                center_mode,
                model.runner.class_names,
            )
            for detection in detections:
                detection["model_name"] = model.name
                detection["pipeline_name"] = runtime.config.name
                detection["model_task"] = model.runner.task
                if model.visualize:
                    detection["visualize"] = model.visualize
            model.last_detections = detections
        merged.extend(model.last_detections)
    return merged


def update_pipeline(runtime: PipelineRuntime, project_root: Path) -> bool:
    pipeline = runtime.config
    ok, packet = runtime.capture.read()
    if not ok or packet is None:
        print(f"[WARN] Pipeline {pipeline.name}: failed to read a frame.")
        return False

    frame = packet.color
    runtime.frame_id += 1
    depth_for_lookup = prepare_depth_for_lookup(
        packet.depth,
        frame.shape,
        packet.depth_aligned_to_color,
    )
    if packet.depth is not None and depth_for_lookup is None and not runtime.warned_depth_unaligned:
        print(
            f"[WARN] Pipeline {pipeline.name}: depth frame is not aligned to RGB; Z lookup is disabled."
        )
        runtime.warned_depth_unaligned = True

    if bool(pipeline.debug.get("preview_only", False)):
        detections: list[dict[str, Any]] = []
        overlays: list[dict[str, Any]] = []
        plugin_status: list[str] = []
        annotated = frame.copy()
    else:
        detections = run_serial_models(runtime, frame, depth_for_lookup)
        processed = runtime.post_processor.process(detections, frame, depth_for_lookup)
        detections = processed["detections"]
        plugin_status = processed["status_lines"]
        overlays = processed.get("overlays", [])
        annotated = draw_detections(frame, detections, pipeline.visualize)
        draw_overlays(annotated, overlays)

    elapsed = max(time.time() - runtime.started_at, 1e-6)
    status_lines = [
        f"{pipeline.name}",
        f"FPS: {runtime.frame_id / elapsed:.1f}",
    ]
    if bool(pipeline.debug.get("preview_only", False)):
        status_lines.append("Preview only")
    else:
        status_lines.append(f"Detections: {len(detections)}")
        status_lines.append("Models: " + ",".join(model.name for model in runtime.models))
    status_lines.extend(plugin_status)
    draw_status_lines(annotated, status_lines[:7])

    if pipeline.debug.get("print_detections", False):
        print_detections(runtime.frame_id, detections)

    save_video_path = resolve_path(project_root, pipeline.output.get("save_video"))
    if save_video_path is not None:
        if runtime.writer is None:
            height, width = annotated.shape[:2]
            fps = runtime.capture.get(cv2.CAP_PROP_FPS) or float(pipeline.camera.get("stream", {}).get("fps", 30.0))
            runtime.writer = create_writer(save_video_path, fps, (width, height))
        runtime.writer.write(annotated)

    if runtime.display is None and bool(pipeline.display.get("enabled", True)):
        try:
            runtime.display = create_display(pipeline.display, pipeline.models[0].inference if pipeline.models else {}, frame.shape)
        except (RuntimeError, cv2.error) as exc:
            print(f"[ERROR] Pipeline {pipeline.name}: display window is unavailable: {exc}")
            return False

    if runtime.display is not None:
        action = runtime.display.update(annotated)
        if action == "quit":
            return False
        if action == "snapshot":
            snapshot_dir = resolve_path(project_root, pipeline.output.get("snapshot_dir"))
            if snapshot_dir is not None:
                path = save_snapshot(snapshot_dir / pipeline.name, annotated)
                print(f"[OK] Pipeline {pipeline.name}: snapshot saved: {path}")
        if action == "reset":
            runtime.post_processor.handle_action("reset")
            print(f"[OK] Pipeline {pipeline.name}: reset requested.")
    return True


def close_pipeline(runtime: PipelineRuntime) -> None:
    runtime.capture.release()
    if runtime.writer is not None:
        runtime.writer.release()
    if runtime.display is not None:
        runtime.display.close()


def run_multi_camera(config: MultiRuntimeConfig) -> None:
    runtimes: list[PipelineRuntime] = []
    for pipeline in config.pipelines:
        try:
            runtimes.append(open_pipeline_runtime(pipeline, config.project_root))
        except Exception as exc:
            print(f"[ERROR] Pipeline {pipeline.name}: startup failed: {exc}")
    if not runtimes:
        raise SystemExit("[ERROR] No pipeline could be started.")

    print("[OK] Press q or Esc in any window to close that pipeline. Press s for snapshot, R to reset postprocess.")
    try:
        while any(runtime.active for runtime in runtimes):
            for runtime in runtimes:
                if not runtime.active:
                    continue
                runtime.active = update_pipeline(runtime, config.project_root)
    finally:
        for runtime in runtimes:
            close_pipeline(runtime)


def main(argv: list[str] | None = None) -> None:
    cli = parse_multi_cli_args(argv)
    if cli.list_cameras:
        list_cameras(DEFAULTS["camera"])
        return

    config = load_multi_runtime_config(cli)
    if cli.dry_run:
        dry_run(config)
        return
    run_multi_camera(config)
