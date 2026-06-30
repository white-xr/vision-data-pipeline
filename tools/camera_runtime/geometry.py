from __future__ import annotations

from typing import Any

import cv2


def lookup_depth_mm(depth_image: Any, center_x: int, center_y: int) -> float | None:
    if depth_image is None:
        return None
    height, width = depth_image.shape[:2]
    if center_x < 0 or center_y < 0 or center_x >= width or center_y >= height:
        return None
    depth_mm = float(depth_image[center_y, center_x])
    if depth_mm <= 0.0:
        return None
    return depth_mm


def mean_depth_for_mask(depth_image: Any, binary_mask: Any) -> float | None:
    if depth_image is None or binary_mask is None:
        return None
    if depth_image.shape[:2] != binary_mask.shape[:2]:
        return None

    import numpy as np

    values = depth_image[(binary_mask > 0) & (depth_image > 0)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def prepare_depth_for_lookup(depth_image: Any, image_shape: tuple[int, ...], aligned_to_color: bool) -> Any:
    if depth_image is None:
        return None

    image_height, image_width = image_shape[:2]
    depth_height, depth_width = depth_image.shape[:2]
    if depth_width == image_width and depth_height == image_height:
        return depth_image
    if not aligned_to_color:
        return None
    return cv2.resize(depth_image, (image_width, image_height), interpolation=cv2.INTER_NEAREST)


def mask_from_polygon(mask_points: Any, image_shape: tuple[int, ...]) -> Any:
    if mask_points is None or len(mask_points) == 0:
        return None

    import numpy as np

    height, width = image_shape[:2]
    points = np.asarray(mask_points, dtype=np.int32)
    if points.size == 0:
        return None
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    if cv2.countNonZero(mask) == 0:
        return None
    return mask


def binary_mask_center(binary_mask: Any, mode: str = "centroid") -> tuple[int, int] | None:
    import numpy as np

    if binary_mask is None:
        return None
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None

    if mode == "bottom":
        max_y = int(ys.max())
        bottom_xs = xs[ys >= max_y - 2]
        return int(round(float(bottom_xs.mean()))), max_y

    moments = cv2.moments(binary_mask, binaryImage=True)
    if moments["m00"] != 0:
        center_x = int(round(moments["m10"] / moments["m00"]))
        center_y = int(round(moments["m01"] / moments["m00"]))
        return center_x, center_y

    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def clean_binary_mask(binary_mask: Any, kernel_size: int, open_iterations: int, close_iterations: int) -> Any:
    if binary_mask is None:
        return None
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return binary_mask
    if kernel_size % 2 == 0:
        kernel_size += 1

    import numpy as np

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = binary_mask
    if open_iterations > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    if close_iterations > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    return cleaned


def box_center(box_xyxy: list[float]) -> tuple[int, int]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
