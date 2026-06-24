"""Run real-time YOLO detection from a camera stream.

Example:
    python tools/camera_detect.py --camera-index 0

Use an RTSP/HTTP stream:
    python tools/camera_detect.py --camera-url rtsp://user:pass@192.168.1.10/stream1
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


DEFAULT_MODEL = Path(
    "runs/detect/data/models/hole_detect_v1/yolo11n_1280_v1/weights/best.pt"
)
CLASS_NAMES = {
    0: "cover_edge_hole",
    1: "base_edge_hole",
}
BACKENDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "obsensor": getattr(cv2, "CAP_OBSENSOR", cv2.CAP_ANY),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO Detect inference from a local camera or stream."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"YOLO model path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Local camera index used when --camera-url is not set. Default: 0.",
    )
    parser.add_argument(
        "--camera-source",
        choices=["opencv", "orbbec"],
        default="opencv",
        help="Camera source type. Use orbbec for Orbbec RGB-D COLOR_SENSOR. Default: opencv.",
    )
    parser.add_argument(
        "--camera-url",
        type=str,
        default=None,
        help="Camera stream URL, for example RTSP/HTTP. Overrides --camera-index.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", *BACKENDS.keys()],
        default="auto",
        help="OpenCV capture backend for local cameras. Windows auto uses dshow. Default: auto.",
    )
    parser.add_argument(
        "--read-retries",
        type=int,
        default=30,
        help="Frame read retries before giving up. Default: 30.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Scan local camera indexes and exit without loading the model.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Show raw camera frames without loading or running YOLO.",
    )
    parser.add_argument(
        "--print-frame-stats",
        action="store_true",
        help="Print raw frame min/max/mean brightness for debugging black frames.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size. Default: 1280.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold. Default: 0.45.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Inference device, for example 0, cpu. Default: 0.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested camera width for local cameras. Default: 1280.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested camera height for local cameras. Default: 720.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Requested camera FPS for local cameras. Default: 30.",
    )
    parser.add_argument(
        "--orbbec-sdk-dir",
        type=Path,
        default=Path("D:/OrbbecSDK_v2"),
        help="Local Orbbec SDK root used to locate DLLs. Default: D:/OrbbecSDK_v2.",
    )
    parser.add_argument(
        "--orbbec-format",
        choices=["RGB", "MJPG", "YUYV", "UYVY", "NV12", "NV21", "I420", "default"],
        default="RGB",
        help="Requested Orbbec color stream format. Default: RGB.",
    )
    parser.add_argument(
        "--orbbec-timeout-ms",
        type=int,
        default=1000,
        help="Orbbec frame wait timeout in milliseconds. Default: 1000.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Maximum detections per frame. Default: 100.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width. Default: 2.",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Optional output video path, for example data/reports/hole_detect_v1/camera.mp4.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("data/reports/hole_detect_v1/camera_snapshots"),
        help="Directory for snapshots saved by pressing s.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open a display window. Useful for headless OpenCV environments.",
    )
    parser.add_argument(
        "--display-backend",
        choices=["auto", "opencv", "tkinter"],
        default="auto",
        help="Display backend for the live window. Default: auto.",
    )
    parser.add_argument(
        "--print-detections",
        action="store_true",
        help="Print detected class, confidence, and center pixel for each frame.",
    )
    return parser.parse_args()


def resolve_backend(name: str, is_url: bool) -> int:
    if is_url:
        return cv2.CAP_ANY if name == "auto" else BACKENDS[name]
    if name == "auto":
        return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    return BACKENDS[name]


def backend_name(api: int) -> str:
    for name, value in BACKENDS.items():
        if value == api:
            return name
    return str(api)


def read_frame_with_retries(
    capture: cv2.VideoCapture,
    retries: int,
    delay_seconds: float = 0.1,
) -> tuple[bool, object | None]:
    for _ in range(max(1, retries)):
        ok, frame = capture.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(delay_seconds)
    return False, None


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    source: str | int = args.camera_url if args.camera_url else args.camera_index
    api = resolve_backend(args.backend, is_url=bool(args.camera_url))
    capture = cv2.VideoCapture(source, api)

    if not args.camera_url:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)

    if not capture.isOpened():
        raise SystemExit(
            f"[ERROR] Cannot open camera source: {source}. "
            "Check camera index, stream URL, permissions, and whether another app is using it."
        )

    ok, _ = read_frame_with_retries(capture, args.read_retries)
    if not ok:
        capture.release()
        raise SystemExit(
            f"[ERROR] Camera opened but no frame was received: {source}, backend={backend_name(api)}.\n"
            "[HINT] Try another index: python tools/camera_detect.py --list-cameras\n"
            "[HINT] Or try: --backend msmf / --backend any / --camera-index 1"
        )

    return capture


def add_orbbec_dll_dirs(sdk_dir: Path) -> None:
    bin_dir = sdk_dir / "bin"
    if not bin_dir.is_dir():
        return

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(bin_dir))

    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def orbbec_frame_to_bgr(frame, ob_format) -> object | None:
    import numpy as np

    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if color_format == ob_format.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if hasattr(ob_format, "BGR") and color_format == ob_format.BGR:
        return np.resize(data, (height, width, 3))
    if color_format == ob_format.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == ob_format.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    if color_format == ob_format.UYVY:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    if color_format == ob_format.NV12:
        image = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12)
    if color_format == ob_format.NV21:
        image = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV21)
    if color_format == ob_format.I420:
        image = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)

    print(f"[WARN] Unsupported Orbbec color format: {color_format}")
    return None


class OrbbecColorCapture:
    def __init__(self, args: argparse.Namespace) -> None:
        add_orbbec_dll_dirs(args.orbbec_sdk_dir)
        try:
            from pyorbbecsdk import Config, OBError, OBFormat, OBSensorType, Pipeline
        except ImportError as exc:
            raise SystemExit(
                "[ERROR] Missing Orbbec Python wrapper.\n"
                "Install it first: python -m pip install --no-deps pyorbbecsdk2\n"
                f"SDK path checked: {args.orbbec_sdk_dir}"
            ) from exc

        self.ob_format = OBFormat
        self.pipeline = Pipeline()
        config = Config()

        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = None
        if args.orbbec_format != "default":
            requested_format = getattr(OBFormat, args.orbbec_format)
            try:
                color_profile = profile_list.get_video_stream_profile(
                    args.width,
                    0,
                    requested_format,
                    int(args.fps),
                )
            except OBError as exc:
                print(f"[WARN] Requested Orbbec color profile is unavailable: {exc}")

        if color_profile is None:
            color_profile = profile_list.get_default_video_stream_profile()

        print(f"[OK] Orbbec color profile: {color_profile}")
        config.enable_stream(color_profile)
        self.pipeline.start(config)
        self.timeout_ms = args.orbbec_timeout_ms
        self.fps = args.fps

    def read(self) -> tuple[bool, object | None]:
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return False, None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return False, None

        image = orbbec_frame_to_bgr(color_frame, self.ob_format)
        return (image is not None), image

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def release(self) -> None:
        self.pipeline.stop()


def load_model(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] Missing dependency: ultralytics. "
            "Activate the vision-data environment and install dependencies first."
        ) from exc

    return YOLO(str(model_path))


def list_cameras(max_index: int = 10) -> None:
    print("[INFO] Scanning local camera indexes...")
    candidates = ["dshow", "msmf", "any"] if platform.system() == "Windows" else ["any"]
    found = False

    for index in range(max_index):
        for backend in candidates:
            api = BACKENDS[backend]
            capture = cv2.VideoCapture(index, api)
            if not capture.isOpened():
                capture.release()
                continue

            ok, frame = read_frame_with_retries(capture, retries=5, delay_seconds=0.05)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()

            if ok and frame is not None:
                actual_height, actual_width = frame.shape[:2]
                print(
                    f"[OK] index={index}, backend={backend}, "
                    f"frame={actual_width}x{actual_height}, requested={width}x{height}, "
                    f"fps={fps:.1f}, {frame_stats_text(frame)}"
                )
                found = True
            else:
                print(f"[WARN] index={index}, backend={backend}, opened but no frame")

    if not found:
        print("[WARN] No readable local camera was found.")


def create_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    fourcc_name = "mp4v" if suffix in {".mp4", ".m4v"} else "XVID"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, frame_size)
    if not writer.isOpened():
        raise SystemExit(f"[ERROR] Cannot create video writer: {path}")
    return writer


def draw_detection_centers(result, image) -> list[dict[str, object]]:
    detections: list[dict[str, object]] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        center_x = int(round((x1 + x2) / 2.0))
        center_y = int(round((y1 + y2) / 2.0))
        class_name = result.names.get(cls_id, CLASS_NAMES.get(cls_id, str(cls_id)))

        cv2.circle(image, (center_x, center_y), 4, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"({center_x},{center_y})",
            (center_x + 6, center_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "center_x": center_x,
                "center_y": center_y,
            }
        )

    return detections


def print_detections(frame_id: int, detections: list[dict[str, object]]) -> None:
    if not detections:
        print(f"[FRAME {frame_id}] no detections")
        return

    items = []
    for det in detections:
        items.append(
            "{class_name} conf={confidence:.3f} center=({center_x},{center_y})".format(**det)
        )
    print(f"[FRAME {frame_id}] " + " | ".join(items))


def save_snapshot(snapshot_dir: Path, image) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = snapshot_dir / f"camera_{timestamp}.jpg"
    cv2.imwrite(str(path), image)
    return path


def frame_stats_text(image) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    min_value, max_value, _, _ = cv2.minMaxLoc(gray)
    mean_value = float(gray.mean())
    return f"min={min_value:.0f} max={max_value:.0f} mean={mean_value:.1f}"


class OpenCvDisplay:
    def __init__(self, title: str) -> None:
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)

    def update(self, image) -> str | None:
        cv2.imshow(self.title, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return "snapshot"
        return None

    def close(self) -> None:
        cv2.destroyWindow(self.title)


class TkinterDisplay:
    def __init__(self, title: str) -> None:
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
        self.root.protocol("WM_DELETE_WINDOW", self._request_quit)
        self.root.bind("<Key>", self._on_key)
        self.label = tk.Label(self.root)
        self.label.pack(fill=tk.BOTH, expand=True)
        self.photo = None
        self.pending_action: str | None = None

    def _request_quit(self) -> None:
        self.pending_action = "quit"

    def _on_key(self, event) -> None:
        key = event.keysym.lower()
        if key in {"q", "escape"}:
            self.pending_action = "quit"
        elif key == "s":
            self.pending_action = "snapshot"

    def update(self, image) -> str | None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = self.image_module.fromarray(rgb)
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


def create_display(backend: str):
    title = "YOLO Camera Detect"
    if backend == "opencv":
        return OpenCvDisplay(title)
    if backend == "tkinter":
        return TkinterDisplay(title)

    try:
        return OpenCvDisplay(title)
    except cv2.error:
        print("[WARN] OpenCV window is unavailable, falling back to Tkinter display.")
        return TkinterDisplay(title)


def main() -> None:
    args = parse_args()

    if args.list_cameras:
        list_cameras()
        return

    model_path = args.model.resolve()
    model = None
    if not args.preview_only:
        if not model_path.is_file():
            raise SystemExit(f"[ERROR] Model not found: {model_path}")
        model = load_model(model_path)

    if args.camera_source == "orbbec":
        capture = OrbbecColorCapture(args)
    else:
        capture = open_capture(args)

    writer: cv2.VideoWriter | None = None
    display = None
    frame_id = 0
    started_at = time.time()

    if args.preview_only:
        print("[OK] Preview only: YOLO model is not loaded.")
    else:
        print(f"[OK] Model: {model_path}")
    if args.camera_source == "orbbec":
        print(f"[OK] Camera: Orbbec COLOR_SENSOR via SDK {args.orbbec_sdk_dir}")
    else:
        print(f"[OK] Camera: {args.camera_url if args.camera_url else args.camera_index}")
        print(f"[OK] Backend: {backend_name(resolve_backend(args.backend, is_url=bool(args.camera_url)))}")
    print("[OK] Press q or Esc to quit. Press s to save a snapshot.")

    try:
        if not args.no_window:
            try:
                display = create_display(args.display_backend)
            except (RuntimeError, cv2.error) as exc:
                print(f"[ERROR] Display window is unavailable: {exc}")
                print("[HINT] Use --no-window --save-video to run without GUI display.")
                return

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("[WARN] Failed to read a frame from camera.")
                break

            frame_id += 1
            stats_text = frame_stats_text(frame)
            if args.preview_only:
                annotated = frame.copy()
                detections = []
            else:
                result = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    max_det=args.max_det,
                    verbose=False,
                )[0]
                annotated = result.plot(line_width=args.line_width)
                detections = draw_detection_centers(result, annotated)

            elapsed = max(time.time() - started_at, 1e-6)
            fps_text = f"FPS: {frame_id / elapsed:.1f}"
            cv2.putText(
                annotated,
                fps_text,
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                stats_text,
                (12, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.print_frame_stats:
                print(f"[FRAME {frame_id}] {stats_text}")

            if args.print_detections:
                print_detections(frame_id, detections)

            if args.save_video:
                if writer is None:
                    height, width = annotated.shape[:2]
                    fps = capture.get(cv2.CAP_PROP_FPS) or args.fps
                    writer = create_writer(args.save_video, fps, (width, height))
                writer.write(annotated)

            if display is not None:
                try:
                    action = display.update(annotated)
                except RuntimeError as exc:
                    print(f"[ERROR] Display window closed or unavailable: {exc}")
                    break

                if action == "quit":
                    break
                if action == "snapshot":
                    path = save_snapshot(args.snapshot_dir, annotated)
                    print(f"[OK] Snapshot saved: {path}")

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display is not None:
            display.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
