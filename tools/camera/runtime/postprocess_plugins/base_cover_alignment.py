from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

import cv2

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


def clip_box(box_xyxy: list[float], image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box_xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    return x1, y1, x2, y2


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


def contour_candidates(gray_roi: Any, contour_params: dict[str, Any]) -> list[Any]:
    blur_kernel = max(1, int(contour_params.get("blur_kernel", 5)))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    morph_kernel = max(1, int(contour_params.get("morph_kernel", 5)))
    if morph_kernel % 2 == 0:
        morph_kernel += 1

    blurred = cv2.GaussianBlur(gray_roi, (blur_kernel, blur_kernel), 0) if blur_kernel > 1 else gray_roi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    masks = []

    if bool(contour_params.get("use_canny", True)):
        low = int(contour_params.get("canny_low", 40))
        high = int(contour_params.get("canny_high", 120))
        edges = cv2.Canny(blurred, low, high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        edges = cv2.dilate(edges, kernel, iterations=1)
        masks.append(edges)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    masks.append(binary)
    masks.append(cv2.bitwise_not(binary))
    return masks


def maybe_apply_depth_filter(mask: Any, depth_roi: Any, contour_params: dict[str, Any]) -> Any:
    if depth_roi is None or not bool(contour_params.get("use_depth_filter", False)):
        return mask

    import numpy as np

    values = np.asarray(depth_roi, dtype="float32")
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        return mask
    median = float(np.median(valid))
    tolerance = float(contour_params.get("depth_tolerance_mm", 80.0))
    depth_mask = ((values >= median - tolerance) & (values <= median + tolerance)).astype("uint8") * 255
    return cv2.bitwise_and(mask, depth_mask)


def find_main_contour(
    frame: Any,
    depth_image: Any,
    box_xyxy: list[float],
    params: dict[str, Any],
) -> tuple[Any | None, list[dict[str, Any]]]:
    clipped = clip_box(box_xyxy, frame.shape)
    if clipped is None:
        return None, []
    x1, y1, x2, y2 = clipped
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, []

    contour_params = dict(params.get("contour", {}) or {})
    min_area_ratio = float(contour_params.get("min_area_ratio", 0.03))
    min_area = max(8.0, (x2 - x1) * (y2 - y1) * min_area_ratio)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    if float(gray.std()) < float(contour_params.get("min_gray_std", 3.0)):
        return None, []
    depth_roi = None if depth_image is None else depth_image[y1:y2, x1:x2]

    best_contour = None
    best_area = 0.0
    for mask in contour_candidates(gray, contour_params):
        mask = maybe_apply_depth_filter(mask, depth_roi, contour_params)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area <= best_area:
                continue
            best_contour = contour
            best_area = area

    if best_contour is None:
        return None, []

    import numpy as np

    offset = np.array([[[x1, y1]]], dtype=best_contour.dtype)
    global_contour = best_contour + offset
    overlay = {
        "type": "contour",
        "points": global_contour.reshape((-1, 2)).astype(int).tolist(),
        "color": [0, 255, 255],
        "thickness": 2,
    }
    return global_contour, [overlay]


def minrect_from_contour(contour: Any) -> tuple[tuple[float, float] | None, list[dict[str, Any]]]:
    if contour is None or len(contour) < 3:
        return None, []
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (width, height), _ = rect
    if width <= 0 or height <= 0:
        return None, []
    box = cv2.boxPoints(rect)
    overlay = {
        "type": "minrect",
        "points": box.astype(int).tolist(),
        "color": [255, 0, 255],
        "thickness": 2,
    }
    return (float(center_x), float(center_y)), [overlay]


def contour_centroid(contour: Any) -> tuple[float, float] | None:
    if contour is None:
        return None
    moments = cv2.moments(contour)
    if abs(float(moments.get("m00", 0.0))) <= 1e-6:
        return None
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def anchor_center(detection: dict[str, Any], params: dict[str, Any]) -> tuple[float, float] | None:
    anchors = params.get("anchors", {}) or {}
    class_name = norm_name(detection.get("class_name"))
    anchor = None
    for key, value in anchors.items():
        if norm_name(key) == class_name:
            anchor = value
            break
    if not isinstance(anchor, dict):
        return None
    box_xyxy = detection.get("box_xyxy")
    if not box_xyxy:
        return None
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    rx = float(anchor.get("rx", 0.5))
    ry = float(anchor.get("ry", 0.5))
    return x1 + rx * (x2 - x1), y1 + ry * (y2 - y1)


def compute_postprocessed_center(
    detection: dict[str, Any],
    frame: Any,
    depth_image: Any,
    params: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    mode = str(params.get("center_mode", "box")).strip().lower()
    box_xyxy = detection.get("box_xyxy")
    if not box_xyxy:
        return "missing_box", []

    if mode == "box":
        center_x, center_y = box_center(box_xyxy)
        update_center(detection, center_x, center_y, "box", depth_image, params)
        return "box", []

    if mode == "anchor":
        center = anchor_center(detection, params)
        if center is not None:
            update_center(detection, center[0], center[1], "anchor", depth_image, params)
            return "anchor", []
        warn_once(
            f"anchor:{norm_name(detection.get('class_name'))}",
            f"Anchor is not configured for {detection.get('class_name')}; falling back to bbox center.",
        )
        center_x, center_y = box_center(box_xyxy)
        update_center(detection, center_x, center_y, "box_fallback", depth_image, params)
        return "box_fallback", []

    if mode in {"contour", "minrect"}:
        contour, overlays = find_main_contour(frame, depth_image, box_xyxy, params)
        if contour is None:
            warn_once(
                f"contour:{norm_name(detection.get('class_name'))}",
                f"OpenCV contour postprocess failed for {detection.get('class_name')}; falling back to bbox center.",
            )
            center_x, center_y = box_center(box_xyxy)
            update_center(detection, center_x, center_y, "box_fallback", depth_image, params)
            return "box_fallback", overlays

        if mode == "contour":
            center = contour_centroid(contour)
            if center is not None:
                update_center(detection, center[0], center[1], "contour", depth_image, params)
                return "contour", overlays

        rect_center, rect_overlays = minrect_from_contour(contour)
        overlays.extend(rect_overlays)
        if rect_center is not None:
            source = "minrect" if mode == "minrect" else "minrect_fallback"
            update_center(detection, rect_center[0], rect_center[1], source, depth_image, params)
            return source, overlays

        warn_once(
            f"minrect:{norm_name(detection.get('class_name'))}",
            f"minAreaRect postprocess failed for {detection.get('class_name')}; falling back to bbox center.",
        )
        center_x, center_y = box_center(box_xyxy)
        update_center(detection, center_x, center_y, "box_fallback", depth_image, params)
        return "box_fallback", overlays

    warn_once("mode", f"Unknown center_mode={mode!r}; falling back to bbox center.")
    center_x, center_y = box_center(box_xyxy)
    update_center(detection, center_x, center_y, "box_fallback", depth_image, params)
    return "box_fallback", []


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


def process(detections: list[dict[str, Any]], frame: Any, depth_image: Any, params: dict[str, Any]) -> dict[str, Any]:
    base_class = str(params.get("base_class", "Base"))
    cover_class = str(params.get("cover_class", "Cover"))

    processed = [copy_detection(detection) for detection in detections]
    overlays: list[dict[str, Any]] = []
    for detection in processed:
        if is_class(detection, base_class) or is_class(detection, cover_class):
            _, center_overlays = compute_postprocessed_center(detection, frame, depth_image, params)
            overlays.extend(center_overlays)

    base_detection = best_detection(processed, base_class)
    cover_detection = best_detection(processed, cover_class)
    update_base_lock(base_detection, cover_detection, params)

    detected_base_center = center_dict(base_detection)
    cover_center = center_dict(cover_detection)
    locked_base_center = _STATE.locked_base_center
    add_point_overlay(overlays, detected_base_center, "Base detected", [0, 255, 255])
    add_point_overlay(overlays, locked_base_center, "Base LOCKED", [0, 255, 0])
    add_point_overlay(overlays, cover_center, "Cover", [0, 0, 255])

    dx = dy = None
    if locked_base_center is not None and cover_center is not None:
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
