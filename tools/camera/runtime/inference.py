from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .geometry import binary_mask_center, box_center, lookup_depth_mm, mask_from_polygon


def load_class_names(value: Any, project_root: Path) -> dict[int, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {int(key): str(name) for key, name in value.items()}
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}

    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    if not path.is_file():
        raise SystemExit(f"[ERROR] Class names file not found: {path}")
    names: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            name = line.strip()
            if name:
                names[index] = name
    return names


def class_name_for(result: Any, class_id: int, configured_names: dict[int, str]) -> str:
    if class_id in configured_names:
        return configured_names[class_id]
    names = getattr(result, "names", {}) or {}
    if isinstance(names, dict) and class_id in names:
        return str(names[class_id])
    return str(class_id)


def result_mask_points(result: Any, index: int) -> Any:
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return None
    if index >= len(masks.xy):
        return None
    return masks.xy[index]


def detection_center(binary_mask: Any, box_xyxy: list[float], center_mode: str) -> tuple[int, int, str]:
    if binary_mask is not None and center_mode != "box":
        center = binary_mask_center(binary_mask, center_mode)
        if center is not None:
            return center[0], center[1], "mask"
    center_x, center_y = box_center(box_xyxy)
    return center_x, center_y, "box"


class YoloRunner:
    def __init__(self, model_config: dict[str, Any], inference_config: dict[str, Any], project_root: Path) -> None:
        self.model_config = model_config
        self.inference_config = inference_config
        self.project_root = project_root
        self.model_path = self.resolve_model_path(model_config.get("path"))
        self.class_names = load_class_names(model_config.get("class_names"), project_root)
        self.model = None

    def resolve_model_path(self, value: Any) -> Path:
        if value is None:
            raise SystemExit("[ERROR] model.path is required.")
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def validate_model_path(self) -> None:
        if not self.model_path.is_file():
            raise SystemExit(f"[ERROR] Model not found: {self.model_path}")

    def load(self) -> None:
        self.validate_model_path()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                "[ERROR] Missing dependency: ultralytics. "
                "Activate the vision-data environment and install dependencies first."
            ) from exc
        self.model = YOLO(str(self.model_path))

    def predict(self, frame: Any):
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded.")
        device = str(self.inference_config.get("device", "0"))
        use_half = bool(self.inference_config.get("half", False) and device.lower() != "cpu")
        return self.model.predict(
            source=frame,
            imgsz=int(self.inference_config.get("imgsz", 960)),
            conf=float(self.inference_config.get("conf", 0.25)),
            iou=float(self.inference_config.get("iou", 0.45)),
            device=device,
            half=use_half,
            max_det=int(self.inference_config.get("max_det", 100)),
            verbose=False,
        )[0]


def detections_from_result(
    result: Any,
    image_shape: tuple[int, ...],
    depth_image: Any,
    center_mode: str,
    configured_names: dict[int, str],
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for index, box in enumerate(boxes):
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        box_xyxy = [float(value) for value in box.xyxy[0].tolist()]
        mask_points = result_mask_points(result, index)
        binary_mask = mask_from_polygon(mask_points, image_shape)
        area = int(cv2.countNonZero(binary_mask)) if binary_mask is not None else int(
            max(0.0, box_xyxy[2] - box_xyxy[0]) * max(0.0, box_xyxy[3] - box_xyxy[1])
        )
        center_x, center_y, center_source = detection_center(binary_mask, box_xyxy, center_mode)
        depth_mm = lookup_depth_mm(depth_image, center_x, center_y)
        detections.append(
            {
                "index": index,
                "class_id": class_id,
                "class_name": class_name_for(result, class_id, configured_names),
                "confidence": confidence,
                "box_xyxy": box_xyxy,
                "mask_points": mask_points,
                "mask": binary_mask,
                "area": area,
                "center": {
                    "x": center_x,
                    "y": center_y,
                    "source": center_source,
                    "depth_mm": depth_mm,
                },
                "center_x": center_x,
                "center_y": center_y,
                "center_source": center_source,
                "depth_mm": depth_mm,
            }
        )
    return detections
