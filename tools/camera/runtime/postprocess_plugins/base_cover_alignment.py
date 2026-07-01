from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

from runtime.geometry import box_center


WAIT_BASE = "WAIT_BASE"
BASE_LOCKED = "BASE_LOCKED"
ALIGN_COVER = "ALIGN_COVER"


@dataclass
class LockState:
    state: str = WAIT_BASE
    locked_base_center: dict[str, Any] | None = None
    base_stable_count: int = 0
    base_center_history: list[dict[str, Any]] = field(default_factory=list)


_STATE = LockState()
_WARNED: set[str] = set()


def reset() -> None:
    _STATE.state = WAIT_BASE
    _STATE.locked_base_center = None
    _STATE.base_stable_count = 0
    _STATE.base_center_history.clear()
    _WARNED.clear()


def handle_action(action: str, params: dict[str, Any]) -> None:
    if action == "reset" and bool(params.get("base_lock", {}).get("enable_manual_reset", True)):
        reset()


def norm_name(value: Any) -> str:
    return str(value or "").strip().lower()


def is_class(detection: dict[str, Any], expected: str) -> bool:
    return norm_name(detection.get("class_name")) == norm_name(expected)


def detection_conf(detection: dict[str, Any]) -> float:
    return float(detection.get("confidence", 0.0))


def copy_detection(detection: dict[str, Any]) -> dict[str, Any]:
    copied = dict(detection)
    copied["center"] = dict(detection.get("center") or {})
    return copied


def warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    print(f"[WARN] {message}")
    _WARNED.add(key)


def median_depth_mm(depth_image: Any, center_x: int, center_y: int, window: int, params: dict[str, Any]) -> float | None:
    if depth_image is None:
        return None

    import numpy as np

    height, width = depth_image.shape[:2]
    half = max(0, int(window) // 2)
    x1 = max(0, int(center_x) - half)
    y1 = max(0, int(center_y) - half)
    x2 = min(width, int(center_x) + half + 1)
    y2 = min(height, int(center_y) + half + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    values = np.asarray(depth_image[y1:y2, x1:x2], dtype="float32")
    min_depth = float(params.get("depth_min_mm", 1.0))
    max_depth = float(params.get("depth_max_mm", 10000.0))
    valid = values[np.isfinite(values) & (values >= min_depth) & (values <= max_depth)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def update_center(
    detection: dict[str, Any],
    center_x: float,
    center_y: float,
    source: str,
    depth_image: Any,
    params: dict[str, Any],
) -> None:
    x = int(round(center_x))
    y = int(round(center_y))
    depth_mm = median_depth_mm(depth_image, x, y, int(params.get("center_depth_window", 15)), params)
    detection["center"] = {"x": x, "y": y, "source": source, "depth_mm": depth_mm}
    detection["center_x"] = x
    detection["center_y"] = y
    detection["center_source"] = source
    detection["depth_mm"] = depth_mm


def anchor_ratios(detection: dict[str, Any], params: dict[str, Any]) -> tuple[float, float] | None:
    anchors = params.get("anchors", {}) or {}
    class_name = norm_name(detection.get("class_name"))
    for key, value in anchors.items():
        if norm_name(key) == class_name and isinstance(value, dict):
            return float(value.get("rx", 0.5)), float(value.get("ry", 0.5))
    return None


def update_anchor_center(detection: dict[str, Any], depth_image: Any, params: dict[str, Any]) -> str:
    box_xyxy = detection.get("box_xyxy")
    if not box_xyxy:
        return "missing_box"

    ratios = anchor_ratios(detection, params)
    if ratios is None:
        warn_once(
            f"anchor:{norm_name(detection.get('class_name'))}",
            f"Anchor is not configured for {detection.get('class_name')}; falling back to bbox center.",
        )
        center_x, center_y = box_center(box_xyxy)
        update_center(detection, center_x, center_y, "box_fallback", depth_image, params)
        return "box_fallback"

    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    rx, ry = ratios
    center_x = x1 + rx * (x2 - x1)
    center_y = y1 + ry * (y2 - y1)
    update_center(detection, center_x, center_y, "anchor", depth_image, params)
    return "anchor"


def best_detection(detections: list[dict[str, Any]], class_name: str) -> dict[str, Any] | None:
    candidates = [detection for detection in detections if is_class(detection, class_name)]
    if not candidates:
        return None
    return max(candidates, key=detection_conf)


def center_dict(detection: dict[str, Any] | None) -> dict[str, Any] | None:
    if detection is None:
        return None
    center = detection.get("center") or {}
    if center.get("x") is None or center.get("y") is None:
        return None
    return {
        "x": int(center["x"]),
        "y": int(center["y"]),
        "depth_mm": center.get("depth_mm"),
        "confidence": detection_conf(detection),
        "source": center.get("source", detection.get("center_source")),
    }


def history_is_stable(history: list[dict[str, Any]], threshold_px: float) -> bool:
    if len(history) < 2:
        return True
    xs = [float(item["x"]) for item in history]
    ys = [float(item["y"]) for item in history]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    return all(hypot(float(item["x"]) - mean_x, float(item["y"]) - mean_y) <= threshold_px for item in history)


def lock_center_from_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    xs = [float(item["x"]) for item in history]
    ys = [float(item["y"]) for item in history]
    depths = [float(item["depth_mm"]) for item in history if item.get("depth_mm") is not None]
    locked = {
        "x": int(round(sum(xs) / len(xs))),
        "y": int(round(sum(ys) / len(ys))),
        "source": "locked_base",
    }
    locked["depth_mm"] = float(np.median(depths)) if depths else None
    return locked


def update_base_lock(base_detection: dict[str, Any] | None, cover_detection: dict[str, Any] | None, params: dict[str, Any]) -> None:
    lock_params = params.get("base_lock", {}) or {}
    stable_frames = max(1, int(lock_params.get("stable_frames", 5)))
    threshold_px = float(lock_params.get("stable_threshold_px", 5.0))
    conf_threshold = float(lock_params.get("conf_threshold", 0.5))

    if _STATE.state == WAIT_BASE:
        base_center = center_dict(base_detection)
        if base_center is None or base_center.get("confidence", 0.0) < conf_threshold:
            _STATE.base_center_history.clear()
            _STATE.base_stable_count = 0
            return

        _STATE.base_center_history.append(base_center)
        _STATE.base_center_history = _STATE.base_center_history[-stable_frames:]
        if history_is_stable(_STATE.base_center_history, threshold_px):
            _STATE.base_stable_count = len(_STATE.base_center_history)
        else:
            _STATE.base_center_history = [base_center]
            _STATE.base_stable_count = 1

        if _STATE.base_stable_count >= stable_frames:
            _STATE.locked_base_center = lock_center_from_history(_STATE.base_center_history)
            _STATE.state = BASE_LOCKED

    if _STATE.state in {BASE_LOCKED, ALIGN_COVER} and _STATE.locked_base_center is not None:
        if cover_detection is not None and center_dict(cover_detection) is not None:
            _STATE.state = ALIGN_COVER


def add_point_overlay(overlays: list[dict[str, Any]], center: dict[str, Any] | None, label: str, color: list[int]) -> None:
    if center is None:
        return
    overlays.append(
        {
            "type": "point",
            "x": center["x"],
            "y": center["y"],
            "label": label,
            "color": color,
            "radius": 6,
        }
    )


def add_alignment_line(
    overlays: list[dict[str, Any]],
    locked_base_center: dict[str, Any] | None,
    cover_center: dict[str, Any] | None,
) -> tuple[int | None, int | None]:
    if locked_base_center is None or cover_center is None:
        return None, None
    dx = int(cover_center["x"] - locked_base_center["x"])
    dy = int(cover_center["y"] - locked_base_center["y"])
    overlays.append(
        {
            "type": "line",
            "start": [locked_base_center["x"], locked_base_center["y"]],
            "end": [cover_center["x"], cover_center["y"]],
            "color": [255, 255, 255],
            "thickness": 2,
        }
    )
    return dx, dy


def process(detections: list[dict[str, Any]], frame: Any, depth_image: Any, params: dict[str, Any]) -> dict[str, Any]:
    base_class = str(params.get("base_class", "Base"))
    cover_class = str(params.get("cover_class", "Cover"))

    processed = [copy_detection(detection) for detection in detections]
    for detection in processed:
        if is_class(detection, base_class) or is_class(detection, cover_class):
            update_anchor_center(detection, depth_image, params)

    base_detection = best_detection(processed, base_class)
    cover_detection = best_detection(processed, cover_class)
    update_base_lock(base_detection, cover_detection, params)

    detected_base_center = center_dict(base_detection)
    cover_center = center_dict(cover_detection)
    locked_base_center = _STATE.locked_base_center

    overlays: list[dict[str, Any]] = []
    dx, dy = add_alignment_line(overlays, locked_base_center, cover_center)
    if locked_base_center is None:
        add_point_overlay(overlays, detected_base_center, "Base detected", [0, 255, 255])
    add_point_overlay(overlays, locked_base_center, "Base LOCKED", [0, 255, 0])
    add_point_overlay(overlays, cover_center, "Cover", [0, 0, 255])

    status_lines = [
        f"State: {_STATE.state}",
        "base target: LOCKED" if locked_base_center is not None else "base target: WAITING",
    ]
    if _STATE.state == WAIT_BASE:
        status_lines.append(f"base stable: {_STATE.base_stable_count}/{int(params.get('base_lock', {}).get('stable_frames', 5))}")
    if dx is not None and dy is not None:
        status_lines.append(f"dx={dx:+d}px dy={dy:+d}px")
    status_lines.append("R: reset")

    return {"detections": processed, "status_lines": status_lines, "overlays": overlays}
