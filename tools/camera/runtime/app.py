from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from .camera import frame_stats_text, list_cameras, open_camera
from .config import RuntimeConfig, load_runtime_config, parse_cli_args
from .display import create_display
from .geometry import prepare_depth_for_lookup
from .inference import YoloRunner, detections_from_result
from .postprocess import load_post_processor
from .visualize import draw_detections, draw_overlays, draw_status_lines


def create_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    fourcc_name = "mp4v" if suffix in {".mp4", ".m4v"} else "XVID"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, frame_size)
    if not writer.isOpened():
        raise SystemExit(f"[ERROR] Cannot create video writer: {path}")
    return writer


def save_snapshot(snapshot_dir: Path, image: Any) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = snapshot_dir / f"camera_{timestamp}.jpg"
    cv2.imwrite(str(path), image)
    return path


def print_detections(frame_id: int, detections: list[dict[str, Any]]) -> None:
    if not detections:
        print(f"[FRAME {frame_id}] no detections")
        return
    items = []
    for detection in detections:
        center = detection.get("center") or {}
        depth = center.get("depth_mm", detection.get("depth_mm"))
        depth_text = f" z={float(depth):.0f}mm" if depth is not None else " z=?"
        items.append(
            f"{detection.get('class_name', '?')} "
            f"conf={float(detection.get('confidence', 0.0)):.3f} "
            f"center=({center.get('x', detection.get('center_x'))},{center.get('y', detection.get('center_y'))})"
            + depth_text
            + f" source={center.get('source', detection.get('center_source', '?'))}"
        )
    print(f"[FRAME {frame_id}] " + " | ".join(items))


def dry_run(config: RuntimeConfig) -> None:
    runner = YoloRunner(config.model, config.inference, config.project_root)
    if not bool(config.debug.get("preview_only", False)):
        runner.validate_model_path()
    post_processor = load_post_processor(config.postprocess)
    print("[OK] Dry run passed.")
    print(f"[OK] Config: {config.config_path}")
    if config.profile_name:
        print(f"[OK] Profile: {config.profile_name}")
    if config.description:
        print(f"[OK] Description: {config.description}")
    print(f"[OK] Camera source: {config.camera.get('source')}")
    print(f"[OK] Model: {runner.model_path}")
    print(
        "[OK] Inference: "
        f"imgsz={config.inference.get('imgsz')}, conf={config.inference.get('conf')}, "
        f"iou={config.inference.get('iou')}, device={config.inference.get('device')}"
    )
    print(f"[OK] Postprocess: {post_processor.label}")


def run_camera(config: RuntimeConfig) -> None:
    preview_only = bool(config.debug.get("preview_only", False))
    post_processor = load_post_processor(config.postprocess)
    runner = YoloRunner(config.model, config.inference, config.project_root)
    if not preview_only:
        runner.load()

    capture = open_camera(config.camera)
    writer: cv2.VideoWriter | None = None
    display = None
    frame_id = 0
    started_at = time.time()
    warned_depth_unaligned = False
    last_detections: list[dict[str, Any]] = []
    infer_stride = max(1, int(config.inference.get("infer_stride", 1)))
    snapshot_dir = config.resolve_path(config.output.get("snapshot_dir"))
    save_video_path = config.resolve_path(config.output.get("save_video"))

    if preview_only:
        print("[OK] Preview only: model is not loaded.")
    else:
        print(f"[OK] Model: {runner.model_path}")
        print(f"[OK] Inference stride: {infer_stride}")
    if config.profile_name:
        print(f"[OK] Profile: {config.profile_name}")
    print("[OK] Press q or Esc to quit. Press s to save a snapshot.")

    try:
        while True:
            ok, packet = capture.read()
            if not ok or packet is None:
                print("[WARN] Failed to read a frame from camera.")
                break
            frame = packet.color

            if display is None and bool(config.display.get("enabled", True)):
                try:
                    display = create_display(config.display, config.inference, frame.shape)
                except (RuntimeError, cv2.error) as exc:
                    print(f"[ERROR] Display window is unavailable: {exc}")
                    print("[HINT] Use --no-window --save-video to run without GUI display.")
                    return

            frame_id += 1
            depth_for_lookup = prepare_depth_for_lookup(
                packet.depth,
                frame.shape,
                packet.depth_aligned_to_color,
            )
            if packet.depth is not None and depth_for_lookup is None and not warned_depth_unaligned:
                print(
                    "[WARN] Depth frame is not aligned to the RGB image, so Z lookup is disabled. "
                    "Use camera.orbbec.depth.align: sw or hw."
                )
                warned_depth_unaligned = True

            plugin_status: list[str] = []
            overlays: list[dict[str, Any]] = []
            if preview_only:
                detections = []
                annotated = frame.copy()
            else:
                should_infer = frame_id == 1 or (frame_id - 1) % infer_stride == 0
                if should_infer:
                    result = runner.predict(frame)
                    last_detections = detections_from_result(
                        result,
                        frame.shape,
                        depth_for_lookup,
                        str(config.visualize.get("center_mode", "centroid")),
                        runner.class_names,
                    )
                processed = post_processor.process(last_detections, frame, depth_for_lookup)
                detections = processed["detections"]
                plugin_status = processed["status_lines"]
                overlays = processed.get("overlays", [])
                annotated = draw_detections(frame, detections, config.visualize)
                draw_overlays(annotated, overlays)

            elapsed = max(time.time() - started_at, 1e-6)
            status_lines = [f"FPS: {frame_id / elapsed:.1f}"]
            if preview_only:
                status_lines.append("Preview only")
            else:
                status_lines.append(f"Detections: {len(detections)}")
            status_lines.extend(plugin_status)
            draw_status_lines(annotated, status_lines[:7])

            if config.debug.get("print_frame_stats", False):
                print(f"[FRAME {frame_id}] {frame_stats_text(frame)}")
            if config.debug.get("print_detections", False):
                print_detections(frame_id, detections)

            if save_video_path is not None:
                if writer is None:
                    height, width = annotated.shape[:2]
                    fps = capture.get(cv2.CAP_PROP_FPS) or float(config.camera.get("stream", {}).get("fps", 30.0))
                    writer = create_writer(save_video_path, fps, (width, height))
                writer.write(annotated)

            if display is not None:
                try:
                    action = display.update(annotated)
                except RuntimeError as exc:
                    print(f"[ERROR] Display window closed or unavailable: {exc}")
                    break
                if action == "quit":
                    break
                if action == "snapshot" and snapshot_dir is not None:
                    path = save_snapshot(snapshot_dir, annotated)
                    print(f"[OK] Snapshot saved: {path}")
                if action == "reset":
                    post_processor.handle_action("reset")
                    print("[OK] Reset requested.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display is not None:
            display.close()


def main(argv: list[str] | None = None) -> None:
    cli = parse_cli_args(argv)
    config = load_runtime_config(cli)
    if cli.dry_run:
        dry_run(config)
        return
    if cli.list_cameras:
        list_cameras(config.camera)
        return
    run_camera(config)
