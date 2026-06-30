from __future__ import annotations

from typing import Any

import cv2

from camera_runtime.geometry import binary_mask_center, clean_binary_mask, lookup_depth_mm, mean_depth_for_mask


def _depth_text(value: float | None) -> str:
    return f"{value:.0f}mm" if value is not None else "--"


def _update_center(detection: dict[str, Any], center_x: int, center_y: int, source: str, depth_image: Any) -> None:
    depth_mm = lookup_depth_mm(depth_image, center_x, center_y)
    detection["center"] = {
        "x": center_x,
        "y": center_y,
        "source": source,
        "depth_mm": depth_mm,
    }
    detection["center_x"] = center_x
    detection["center_y"] = center_y
    detection["center_source"] = source
    detection["depth_mm"] = depth_mm


def process(detections: list[dict[str, Any]], frame: Any, depth_image: Any, params: dict[str, Any]) -> dict[str, Any]:
    target_class = str(params.get("target_class", "target_plate"))
    suppress_when_target_exists = set(params.get("suppress_when_target_exists", ["screwdriver_tip"]))
    min_area_by_class = dict(params.get("min_area_by_class", {}))
    center_mode = str(params.get("center_mode", "centroid"))
    target_center_mode = str(params.get("target_center_mode", "bottom"))
    morph = params.get("target_morphology", {}) or {}
    kernel_size = int(morph.get("kernel", 3))
    open_iterations = int(morph.get("open", 1))
    close_iterations = int(morph.get("close", 1))

    filtered: list[dict[str, Any]] = []
    target_masks = []
    target_confs = []
    status_lines: list[str] = []

    for detection in detections:
        class_name = str(detection.get("class_name", ""))
        min_area = int(min_area_by_class.get(class_name, 0))
        if min_area > 0 and int(detection.get("area", 0)) < min_area:
            continue
        if class_name == target_class and detection.get("mask") is not None:
            target_masks.append(detection["mask"])
            target_confs.append(float(detection.get("confidence", 0.0)))
            continue
        filtered.append(detection)

    target_exists = bool(target_masks)
    output: list[dict[str, Any]] = []
    if target_exists:
        import numpy as np

        merged_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for mask in target_masks:
            merged_mask = cv2.bitwise_or(merged_mask, mask)
        merged_mask = clean_binary_mask(merged_mask, kernel_size, open_iterations, close_iterations)
        center = binary_mask_center(merged_mask, target_center_mode)
        target_detection = {
            "index": None,
            "class_id": None,
            "class_name": target_class,
            "confidence": max(target_confs) if target_confs else 0.0,
            "box_xyxy": None,
            "mask_points": None,
            "mask": merged_mask,
            "area": int(cv2.countNonZero(merged_mask)),
        }
        if center is not None:
            _update_center(target_detection, center[0], center[1], "merged_mask", depth_image)
        output.append(target_detection)
        mean_depth = mean_depth_for_mask(depth_image, merged_mask)
        status_lines.append(f"{target_class}: {_depth_text(mean_depth)}")

    for detection in filtered:
        class_name = str(detection.get("class_name", ""))
        if target_exists and class_name in suppress_when_target_exists:
            continue
        mask = detection.get("mask")
        if mask is not None and center_mode != "box":
            center = binary_mask_center(mask, center_mode)
            if center is not None:
                _update_center(detection, center[0], center[1], "mask", depth_image)
        output.append(detection)

    return {"detections": output, "status_lines": status_lines}
