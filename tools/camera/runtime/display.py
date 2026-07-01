from __future__ import annotations

from typing import Any

import cv2


def int_or_auto_value(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"auto", "none", "null"}:
        return None
    return int(value)


class OpenCvDisplay:
    def __init__(self, title: str, width: int, height: int) -> None:
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, int(width), int(height))

    def update(self, image: Any) -> str | None:
        cv2.imshow(self.title, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return "snapshot"
        if key in (ord("r"), ord("R")):
            return "reset"
        return None

    def close(self) -> None:
        cv2.destroyWindow(self.title)


class TkinterDisplay:
    def __init__(self, title: str, width: int, height: int, resize_to_window: bool) -> None:
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except ImportError as exc:
            raise RuntimeError(
                "Tkinter display requires tkinter and pillow. Install pillow or use --no-window."
            ) from exc

        self.tk = tk
        self.image_module = Image
        self.image_tk_module = ImageTk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.resize_to_window = resize_to_window
        self.root.protocol("WM_DELETE_WINDOW", self._request_quit)
        self.root.bind("<Key>", self._on_key)
        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill=tk.BOTH, expand=True)
        self.photo = None
        self.pending_action: str | None = None

    def _request_quit(self) -> None:
        self.pending_action = "quit"

    def _on_key(self, event: Any) -> None:
        key = event.keysym.lower()
        if key in {"q", "escape"}:
            self.pending_action = "quit"
        elif key == "s":
            self.pending_action = "snapshot"
        elif key == "r":
            self.pending_action = "reset"

    def update(self, image: Any) -> str | None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = self.image_module.fromarray(rgb)
        if self.resize_to_window:
            target_width = max(1, self.label.winfo_width())
            target_height = max(1, self.label.winfo_height())
            source_width, source_height = pil_image.size
            scale = min(target_width / source_width, target_height / source_height)
            resized_size = (
                max(1, int(source_width * scale)),
                max(1, int(source_height * scale)),
            )
            pil_image = pil_image.resize(resized_size, self.image_module.Resampling.BILINEAR)
        self.photo = self.image_tk_module.PhotoImage(image=pil_image)
        self.label.configure(image=self.photo)
        self.root.update_idletasks()
        self.root.update()
        action = self.pending_action
        self.pending_action = None
        return action

    def close(self) -> None:
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass


def resolve_window_size(display_config: dict[str, Any], frame_shape: tuple[int, ...], inference_config: dict[str, Any]) -> tuple[int, int]:
    image_height, image_width = frame_shape[:2]
    window_width = int_or_auto_value(display_config.get("window_width"))
    window_height = int_or_auto_value(display_config.get("window_height"))

    if window_width is None:
        width = int(inference_config.get("imgsz", image_width)) if display_config.get("resize_to_window") else image_width
    else:
        width = window_width

    if window_height is None:
        height = int(round(width * image_height / max(1, image_width)))
    else:
        height = window_height
    return max(1, int(width)), max(1, int(height))


def create_display(display_config: dict[str, Any], inference_config: dict[str, Any], frame_shape: tuple[int, ...]):
    title = str(display_config.get("window_title", "YOLO Camera Inference"))
    backend = str(display_config.get("backend", "auto"))
    window_width, window_height = resolve_window_size(display_config, frame_shape, inference_config)
    if backend == "opencv":
        return OpenCvDisplay(title, window_width, window_height)
    if backend == "tkinter":
        return TkinterDisplay(title, window_width, window_height, bool(display_config.get("resize_to_window", True)))

    try:
        return OpenCvDisplay(title, window_width, window_height)
    except cv2.error:
        print("[WARN] OpenCV window is unavailable, falling back to Tkinter display.")
        return TkinterDisplay(title, window_width, window_height, bool(display_config.get("resize_to_window", True)))
