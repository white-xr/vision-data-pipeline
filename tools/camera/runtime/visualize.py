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


def clamp_text_origin(
    image: Any,
    text: str,
    origin: tuple[int, int],
    scale: float,
    thickness: int,
    margin: int = 6,
) -> tuple[int, int]:
    height, width = image.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(margin, min(int(origin[0]), width - text_width - margin))
    y_min = text_height + margin
    y_max = height - baseline - margin
    y = max(y_min, min(int(origin[1]), y_max))
    return x, y


def draw_readable_text(
    image: Any,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
    background: bool = False,
) -> None:
    x, y = clamp_text_origin(image, text, origin, scale, thickness)
    if background:
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(
            image,
            (x - 4, y - text_height - 4),
            (x + text_width + 4, y + baseline + 4),
            (0, 0, 0),
            -1,
        )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        max(thickness + 2, 2),
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_center_marker(
    image: Any,
    center_x: int,
    center_y: int,
    depth_mm: float | None = None,
    show_label: bool = True,
) -> None:
    depth_text = f",Z={depth_mm:.0f}mm" if depth_mm is not None else ",Z=?"
    cv2.circle(image, (center_x, center_y), 4, (0, 0, 255), -1)
    if show_label:
        draw_readable_text(
            image,
            f"({center_x},{center_y}{depth_text})",
            (center_x + 6, center_y - 6),
            (0, 0, 255),
            0.45,
            1,
        )


def detection_visualize_config(default_config: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any]:
    model_config = detection.get("visualize")
    if not isinstance(model_config, dict):
        return default_config
    merged = dict(default_config)
    merged.update(model_config)
    return merged


def draw_detections(image: Any, detections: list[dict[str, Any]], visualize_config: dict[str, Any]):
    annotated = image.copy()

    for detection in detections:
        visual = detection_visualize_config(visualize_config, detection)
        if not visual.get("draw_masks", True):
            continue
        mask = detection.get("mask")
        if mask is None:
            continue
        mask_alpha = float(visual.get("mask_alpha", 0.35))
        color = instance_color(detection.get("class_id"))
        colored = annotated.copy()
        colored[mask > 0] = color
        annotated = cv2.addWeighted(colored, mask_alpha, annotated, 1.0 - mask_alpha, 0)

    default_line_width = normalize_line_width(visualize_config.get("line_width", 2))

    for detection in detections:
        visual = detection_visualize_config(visualize_config, detection)
        line_width = normalize_line_width(visual.get("line_width", default_line_width))
        draw_boxes = bool(visual.get("draw_boxes", True))
        draw_boxes_when_no_mask = bool(visual.get("draw_boxes_when_no_mask", True))
        draw_labels = bool(visual.get("draw_labels", True))
        draw_centers = bool(visual.get("draw_centers", True))
        draw_center_labels = bool(visual.get("draw_center_labels", True))
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
                draw_readable_text(
                    annotated,
                    label,
                    (x1, y1 - 6),
                    color,
                    0.5,
                    max(1, line_width),
                )

        center = detection.get("center") or {}
        center_x = center.get("x", detection.get("center_x"))
        center_y = center.get("y", detection.get("center_y"))
        if draw_centers and center_x is not None and center_y is not None:
            draw_center_marker(
                annotated,
                int(center_x),
                int(center_y),
                center.get("depth_mm", detection.get("depth_mm")),
                draw_center_labels,
            )

    return annotated


def color_tuple(value: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return int(value[0]), int(value[1]), int(value[2])
    return fallback


def draw_overlays(image: Any, overlays: list[dict[str, Any]]) -> None:
    for overlay in overlays:
        kind = overlay.get("type")
        color = color_tuple(overlay.get("color"), (0, 255, 255))
        thickness = int(overlay.get("thickness", 2))

        if kind in {"contour", "polygon", "minrect"}:
            import numpy as np

            points = overlay.get("points")
            if not points:
                continue
            pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        elif kind == "point":
            x = overlay.get("x")
            y = overlay.get("y")
            if x is None or y is None:
                continue
            radius = int(overlay.get("radius", 5))
            cv2.circle(image, (int(round(x)), int(round(y))), radius, color, -1)
            label = overlay.get("label")
            if label:
                draw_readable_text(
                    image,
                    str(label),
                    (int(round(x)) + 8, int(round(y)) - 8),
                    color,
                    float(overlay.get("scale", 0.48)),
                    int(overlay.get("thickness", 1)),
                )
        elif kind == "line":
            start = overlay.get("start")
            end = overlay.get("end")
            if not start or not end:
                continue
            cv2.line(
                image,
                (int(round(start[0])), int(round(start[1]))),
                (int(round(end[0])), int(round(end[1]))),
                color,
                thickness,
                cv2.LINE_AA,
            )
        elif kind == "text":
            text = overlay.get("text")
            at = overlay.get("at", [12, 32])
            if not text:
                continue
            draw_readable_text(
                image,
                str(text),
                (int(round(at[0])), int(round(at[1]))),
                color,
                float(overlay.get("scale", 0.55)),
                thickness,
            )


def draw_status_lines(image: Any, lines: list[str]) -> None:
    for index, line in enumerate(lines):
        y = 24 + index * 22
        draw_readable_text(
            image,
            line,
            (12, y),
            (0, 255, 255),
            0.58,
            1,
            True,
        )
