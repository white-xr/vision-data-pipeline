from __future__ import annotations

from typing import Any

import cv2


PALETTE = [
    (255, 128, 0),
    (0, 220, 255),
    (80, 180, 80),
    (255, 80, 180),
    (180, 80, 255),
    (255, 220, 90),
    (90, 180, 255),
]


def instance_color(class_id: int | None) -> tuple[int, int, int]:
    if class_id is None:
        return (0, 0, 255)
    return PALETTE[int(class_id) % len(PALETTE)]


def normalize_line_width(line_width: Any) -> int:
    return max(1, int(round(float(line_width))))


def draw_center_marker(image: Any, center_x: int, center_y: int, depth_mm: float | None = None) -> None:
    depth_text = f",Z={depth_mm:.0f}mm" if depth_mm is not None else ",Z=?"
    cv2.circle(image, (center_x, center_y), 4, (0, 0, 255), -1)
    cv2.putText(
        image,
        f"({center_x},{center_y}{depth_text})",
        (center_x + 6, center_y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


def draw_detections(image: Any, detections: list[dict[str, Any]], visualize_config: dict[str, Any]):
    annotated = image.copy()
    mask_alpha = float(visualize_config.get("mask_alpha", 0.35))

    if visualize_config.get("draw_masks", True):
        for detection in detections:
            mask = detection.get("mask")
            if mask is None:
                continue
            color = instance_color(detection.get("class_id"))
            colored = annotated.copy()
            colored[mask > 0] = color
            annotated = cv2.addWeighted(colored, mask_alpha, annotated, 1.0 - mask_alpha, 0)

    line_width = normalize_line_width(visualize_config.get("line_width", 2))
    draw_boxes = bool(visualize_config.get("draw_boxes", True))
    draw_boxes_when_no_mask = bool(visualize_config.get("draw_boxes_when_no_mask", True))
    draw_labels = bool(visualize_config.get("draw_labels", True))
    draw_centers = bool(visualize_config.get("draw_centers", True))

    for detection in detections:
        box = detection.get("box_xyxy")
        mask = detection.get("mask")
        color = instance_color(detection.get("class_id"))
        should_draw_box = draw_boxes or (draw_boxes_when_no_mask and mask is None)
        if box is not None and (should_draw_box or draw_labels):
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            if should_draw_box:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)
            if draw_labels:
                label = f"{detection.get('class_name', '?')} {float(detection.get('confidence', 0.0)):.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    max(1, line_width),
                    cv2.LINE_AA,
                )

        center = detection.get("center") or {}
        center_x = center.get("x", detection.get("center_x"))
        center_y = center.get("y", detection.get("center_y"))
        if draw_centers and center_x is not None and center_y is not None:
            draw_center_marker(annotated, int(center_x), int(center_y), center.get("depth_mm", detection.get("depth_mm")))

    return annotated


def draw_status_lines(image: Any, lines: list[str]) -> None:
    for index, line in enumerate(lines):
        y = 32 + index * 28
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
