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
DEFAULT_CONFIG = Path("configs/camera_detect.yaml")
CLASS_NAMES = {
    0: "cover_edge_hole",
    1: "base_edge_hole",
}
TARGET_PLATE_CLASS = "target_plate"
SUPPRESSED_WHEN_TARGET_EXISTS = {"screwdriver_tip"}
BACKENDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "obsensor": getattr(cv2, "CAP_OBSENSOR", cv2.CAP_ANY),
}


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def int_or_auto(value) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"auto", "none", "null"}:
        return None
    return int(value)


ARG_SPECS = {
    "model": {"flags": ["--model"], "type": Path, "default": DEFAULT_MODEL},
    "camera_index": {"flags": ["--camera-index"], "type": int, "default": 0},
    "camera_source": {"flags": ["--camera-source"], "choices": ["opencv", "orbbec"], "default": "opencv"},
    "camera_url": {"flags": ["--camera-url"], "type": str, "default": None},
    "backend": {"flags": ["--backend"], "choices": ["auto", *BACKENDS.keys()], "default": "auto"},
    "read_retries": {"flags": ["--read-retries"], "type": int, "default": 30},
    "opencv_auto_exposure": {"flags": ["--opencv-auto-exposure"], "type": str_to_bool, "default": None},
    "opencv_exposure": {"flags": ["--opencv-exposure"], "type": float, "default": None},
    "opencv_gain": {"flags": ["--opencv-gain"], "type": float, "default": None},
    "list_cameras": {"flags": ["--list-cameras"], "action": "store_true", "default": False},
    "preview_only": {"flags": ["--preview-only"], "action": "store_true", "default": False},
    "print_frame_stats": {"flags": ["--print-frame-stats"], "action": "store_true", "default": False},
    "imgsz": {"flags": ["--imgsz"], "type": int, "default": 1280},
    "conf": {"flags": ["--conf"], "type": float, "default": 0.25},
    "iou": {"flags": ["--iou"], "type": float, "default": 0.45},
    "device": {"flags": ["--device"], "default": "0"},
    "width": {"flags": ["--width"], "type": int, "default": 1280},
    "height": {"flags": ["--height"], "type": int, "default": 720},
    "fps": {"flags": ["--fps"], "type": float, "default": 30.0},
    "orbbec_sdk_dir": {"flags": ["--orbbec-sdk-dir"], "type": Path, "default": Path("D:/OrbbecSDK_v2")},
    "orbbec_format": {
        "flags": ["--orbbec-format"],
        "choices": ["RGB", "MJPG", "YUYV", "UYVY", "NV12", "NV21", "I420", "default"],
        "default": "RGB",
    },
    "orbbec_depth_align": {"flags": ["--orbbec-depth-align"], "choices": ["sw", "hw", "disable"], "default": "sw"},
    "disable_depth": {"flags": ["--disable-depth"], "action": "store_true", "default": False},
    "orbbec_timeout_ms": {"flags": ["--orbbec-timeout-ms"], "type": int, "default": 1000},
    "orbbec_color_auto_exposure": {"flags": ["--orbbec-color-auto-exposure"], "type": str_to_bool, "default": None},
    "orbbec_color_exposure": {"flags": ["--orbbec-color-exposure"], "type": int, "default": None},
    "orbbec_color_gain": {"flags": ["--orbbec-color-gain"], "type": int, "default": None},
    "orbbec_color_ae_max_exposure": {"flags": ["--orbbec-color-ae-max-exposure"], "type": int, "default": None},
    "orbbec_color_ae_max_gain": {"flags": ["--orbbec-color-ae-max-gain"], "type": int, "default": None},
    "orbbec_depth_auto_exposure": {"flags": ["--orbbec-depth-auto-exposure"], "type": str_to_bool, "default": None},
    "orbbec_depth_exposure": {"flags": ["--orbbec-depth-exposure"], "type": int, "default": None},
    "orbbec_depth_gain": {"flags": ["--orbbec-depth-gain"], "type": int, "default": None},
    "orbbec_ir_auto_exposure": {"flags": ["--orbbec-ir-auto-exposure"], "type": str_to_bool, "default": None},
    "orbbec_ir_exposure": {"flags": ["--orbbec-ir-exposure"], "type": int, "default": None},
    "orbbec_ir_gain": {"flags": ["--orbbec-ir-gain"], "type": int, "default": None},
    "max_det": {"flags": ["--max-det"], "type": int, "default": 100},
    "line_width": {"flags": ["--line-width"], "type": float, "default": 2.0},
    "draw_masks": {"flags": ["--draw-masks"], "type": str_to_bool, "default": True},
    "draw_boxes": {"flags": ["--draw-boxes"], "type": str_to_bool, "default": True},
    "draw_labels": {"flags": ["--draw-labels"], "type": str_to_bool, "default": True},
    "conf_thres": {"flags": ["--conf-thres"], "type": float, "default": 0.35},
    "target_plate_min_area": {"flags": ["--target-plate-min-area"], "type": int, "default": 200},
    "screwdriver_tip_min_area": {"flags": ["--screwdriver-tip-min-area"], "type": int, "default": 100},
    "target_plate_morph_kernel": {"flags": ["--target-plate-morph-kernel"], "type": int, "default": 3},
    "target_plate_morph_open": {"flags": ["--target-plate-morph-open"], "type": int, "default": 1},
    "target_plate_morph_close": {"flags": ["--target-plate-morph-close"], "type": int, "default": 1},
    "mask_center_mode": {
        "flags": ["--mask-center-mode"],
        "choices": ["centroid", "bottom", "box"],
        "default": "centroid",
    },
    "save_video": {"flags": ["--save-video"], "type": Path, "default": None},
    "snapshot_dir": {
        "flags": ["--snapshot-dir"],
        "type": Path,
        "default": Path("data/reports/hole_detect_v1/camera_snapshots"),
    },
    "no_window": {"flags": ["--no-window"], "action": "store_true", "default": False},
    "display_backend": {"flags": ["--display-backend"], "choices": ["auto", "opencv", "tkinter"], "default": "auto"},
    "window_width": {"flags": ["--window-width"], "type": int_or_auto, "default": 1280},
    "window_height": {"flags": ["--window-height"], "type": int_or_auto, "default": 720},
    "resize_to_window": {"flags": ["--resize-to-window"], "type": str_to_bool, "default": True},
    "print_detections": {"flags": ["--print-detections"], "action": "store_true", "default": False},
}


def add_runtime_arguments(parser: argparse.ArgumentParser, use_defaults: bool) -> None:
    for dest, spec in ARG_SPECS.items():
        kwargs = {key: value for key, value in spec.items() if key != "flags"}
        if not use_defaults:
            kwargs["default"] = None
        else:
            kwargs.setdefault("default", spec.get("default"))
        parser.add_argument(*spec["flags"], dest=dest, **kwargs)


def parse_cli_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Run YOLO Detect inference from a local camera or stream."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Ignore YAML config and use built-in defaults plus command line arguments.",
    )
    add_runtime_arguments(parser, use_defaults=False)

    defaults_parser = argparse.ArgumentParser(add_help=False)
    defaults_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    defaults_parser.add_argument("--no-config", action="store_true")
    add_runtime_arguments(defaults_parser, use_defaults=True)
    return parser.parse_args(argv), defaults_parser.parse_args(argv)


def load_yaml_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        if path == DEFAULT_CONFIG:
            return {}
        raise SystemExit(f"[ERROR] Config file not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] Missing dependency: PyYAML. Install dependencies first: python -m pip install PyYAML"
        ) from exc

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise SystemExit(f"[ERROR] Config file must contain a YAML mapping: {path}")
    return data


def coerce_config_value(dest: str, value):
    if value is None:
        return None
    value_type = ARG_SPECS[dest].get("type")
    if value_type is Path:
        return Path(value)
    if value_type is not None:
        return value_type(value)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_args, defaults = parse_cli_args(argv)
    values = vars(defaults).copy()

    if not cli_args.no_config:
        config = load_yaml_config(cli_args.config)
        unknown_keys = sorted(set(config) - set(ARG_SPECS))
        if unknown_keys:
            raise SystemExit(f"[ERROR] Unknown config keys in {cli_args.config}: {', '.join(unknown_keys)}")
        for key, value in config.items():
            values[key] = coerce_config_value(key, value)

    for key, value in vars(cli_args).items():
        if key in {"config", "no_config"}:
            values[key] = value
        elif value is not None:
            values[key] = value

    values["config"] = cli_args.config
    values["no_config"] = cli_args.no_config
    return argparse.Namespace(**values)


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


def maybe_set_opencv_property(capture: cv2.VideoCapture, prop: int, value, name: str) -> None:
    if value is None:
        return
    ok = capture.set(prop, value)
    actual = capture.get(prop)
    if ok:
        print(f"[OK] OpenCV {name}: requested={value}, actual={actual}")
    else:
        print(f"[WARN] OpenCV {name} is not supported by this camera/backend.")


def apply_opencv_camera_controls(capture: cv2.VideoCapture, args: argparse.Namespace) -> None:
    if args.opencv_auto_exposure is not None:
        # DirectShow commonly uses 0.75 for auto and 0.25 for manual exposure.
        auto_value = 0.75 if args.opencv_auto_exposure else 0.25
        maybe_set_opencv_property(capture, cv2.CAP_PROP_AUTO_EXPOSURE, auto_value, "auto_exposure")
    maybe_set_opencv_property(capture, cv2.CAP_PROP_EXPOSURE, args.opencv_exposure, "exposure")
    maybe_set_opencv_property(capture, cv2.CAP_PROP_GAIN, args.opencv_gain, "gain")


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    source: str | int = args.camera_url if args.camera_url else args.camera_index
    api = resolve_backend(args.backend, is_url=bool(args.camera_url))
    capture = cv2.VideoCapture(source, api)

    if not args.camera_url:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        apply_opencv_camera_controls(capture, args)

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


def orbbec_depth_frame_to_mm(depth_frame):
    import numpy as np

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16, count=width * height)
    return data.reshape((height, width)).astype("float32") * float(scale)


def orbbec_property_supported(device, property_id, permission_name: str = "write") -> bool:
    try:
        supported = device.is_property_supported(property_id, permission_name)
        return bool(supported)
    except TypeError:
        try:
            return bool(device.is_property_supported(property_id))
        except Exception:
            return False
    except Exception:
        return False


def set_orbbec_bool_property(device, ob_property_id, name: str, value) -> None:
    if value is None:
        return
    property_id = getattr(ob_property_id, name)
    try:
        device.set_bool_property(property_id, bool(value))
        actual = device.get_bool_property(property_id)
        print(f"[OK] Orbbec {name}: requested={bool(value)}, actual={actual}")
    except Exception as exc:
        print(f"[WARN] Orbbec {name} is not supported or failed to set: {exc}")


def set_orbbec_int_property(device, ob_property_id, name: str, value) -> None:
    if value is None:
        return
    property_id = getattr(ob_property_id, name)
    try:
        device.set_int_property(property_id, int(value))
        actual = device.get_int_property(property_id)
        print(f"[OK] Orbbec {name}: requested={int(value)}, actual={actual}")
    except Exception as exc:
        print(f"[WARN] Orbbec {name} is not supported or failed to set: {exc}")


def apply_orbbec_camera_controls(device, ob_property_id, args: argparse.Namespace) -> None:
    set_orbbec_bool_property(
        device,
        ob_property_id,
        "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL",
        args.orbbec_color_auto_exposure,
    )
    set_orbbec_int_property(
        device,
        ob_property_id,
        "OB_PROP_COLOR_AE_MAX_EXPOSURE_INT",
        args.orbbec_color_ae_max_exposure,
    )
    set_orbbec_int_property(
        device,
        ob_property_id,
        "OB_PROP_COLOR_AE_MAX_GAIN_INT",
        args.orbbec_color_ae_max_gain,
    )
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_COLOR_EXPOSURE_INT", args.orbbec_color_exposure)
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_COLOR_GAIN_INT", args.orbbec_color_gain)

    set_orbbec_bool_property(
        device,
        ob_property_id,
        "OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL",
        args.orbbec_depth_auto_exposure,
    )
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_DEPTH_EXPOSURE_INT", args.orbbec_depth_exposure)
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_DEPTH_GAIN_INT", args.orbbec_depth_gain)

    set_orbbec_bool_property(device, ob_property_id, "OB_PROP_IR_AUTO_EXPOSURE_BOOL", args.orbbec_ir_auto_exposure)
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_IR_EXPOSURE_INT", args.orbbec_ir_exposure)
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_IR_GAIN_INT", args.orbbec_ir_gain)


class OrbbecColorCapture:
    def __init__(self, args: argparse.Namespace) -> None:
        add_orbbec_dll_dirs(args.orbbec_sdk_dir)
        try:
            from pyorbbecsdk import (
                Config,
                OBAlignMode,
                OBError,
                OBFormat,
                OBFrameAggregateOutputMode,
                OBPropertyID,
                OBSensorType,
                Pipeline,
            )
        except ImportError as exc:
            raise SystemExit(
                "[ERROR] Missing Orbbec Python wrapper.\n"
                "Install it first: python -m pip install --no-deps pyorbbecsdk2\n"
                f"SDK path checked: {args.orbbec_sdk_dir}"
            ) from exc

        self.ob_format = OBFormat
        self.pipeline = Pipeline()
        self.enable_depth = not args.disable_depth
        self.last_depth_image = None
        self.depth_aligned_to_color = False
        config = Config()
        apply_orbbec_camera_controls(self.pipeline.get_device(), OBPropertyID, args)

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

        if self.enable_depth:
            depth_profile = None
            try:
                depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = depth_profiles.get_default_video_stream_profile()
                print(f"[OK] Orbbec depth profile: {depth_profile}")
                config.enable_stream(depth_profile)
                config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

                if args.orbbec_depth_align == "hw":
                    config.set_align_mode(OBAlignMode.HW_MODE)
                    self.depth_aligned_to_color = True
                    print("[OK] Orbbec depth align: hardware depth-to-color")
                elif args.orbbec_depth_align == "sw":
                    config.set_align_mode(OBAlignMode.SW_MODE)
                    self.depth_aligned_to_color = True
                    print("[OK] Orbbec depth align: software depth-to-color")
                else:
                    config.set_align_mode(OBAlignMode.DISABLE)
                    print("[WARN] Orbbec depth align disabled. Z lookup may not match RGB pixels.")
            except Exception as exc:
                self.enable_depth = False
                print(f"[WARN] Orbbec depth stream is unavailable, running RGB only: {exc}")

        if self.enable_depth:
            try:
                self.pipeline.enable_frame_sync()
            except Exception as exc:
                print(f"[WARN] Orbbec frame sync could not be enabled: {exc}")

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
        self.last_depth_image = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                self.last_depth_image = orbbec_depth_frame_to_mm(depth_frame)

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


def lookup_depth_mm(depth_image, center_x: int, center_y: int) -> float | None:
    if depth_image is None:
        return None
    height, width = depth_image.shape[:2]
    if center_x < 0 or center_y < 0 or center_x >= width or center_y >= height:
        return None

    depth_mm = float(depth_image[center_y, center_x])
    if depth_mm <= 0.0:
        return None
    return depth_mm


def prepare_depth_for_lookup(depth_image, image_shape, aligned_to_color: bool):
    if depth_image is None:
        return None

    image_height, image_width = image_shape[:2]
    depth_height, depth_width = depth_image.shape[:2]
    if depth_width == image_width and depth_height == image_height:
        return depth_image

    if not aligned_to_color:
        return None

    return cv2.resize(depth_image, (image_width, image_height), interpolation=cv2.INTER_NEAREST)


def mask_center(mask_points, mode: str) -> tuple[int, int] | None:
    if mask_points is None or len(mask_points) == 0:
        return None

    import numpy as np

    points = np.asarray(mask_points, dtype="float32")
    if mode == "bottom":
        max_y = points[:, 1].max()
        bottom_points = points[points[:, 1] >= max_y - 2.0]
        center_x = int(round(float(bottom_points[:, 0].mean())))
        center_y = int(round(float(max_y)))
        return center_x, center_y

    moments = cv2.moments(points)
    if moments["m00"] != 0:
        center_x = int(round(moments["m10"] / moments["m00"]))
        center_y = int(round(moments["m01"] / moments["m00"]))
        return center_x, center_y

    center_x = int(round(float(points[:, 0].mean())))
    center_y = int(round(float(points[:, 1].mean())))
    return center_x, center_y


def result_mask_points(result, index: int):
    if result.masks is None or result.masks.xy is None or index >= len(result.masks.xy):
        return None
    return result.masks.xy[index]


def normalize_line_width(line_width: float) -> int:
    return max(1, int(round(line_width)))


def class_name_for_box(result, cls_id: int) -> str:
    return result.names.get(cls_id, CLASS_NAMES.get(cls_id, str(cls_id)))


def merged_mask_from_polygons(mask_polygons, image_shape):
    if not mask_polygons:
        return None

    import numpy as np

    height, width = image_shape[:2]
    merged = np.zeros((height, width), dtype=np.uint8)
    for polygon in mask_polygons:
        points = np.asarray(polygon, dtype=np.int32)
        if points.size == 0:
            continue
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(merged, [points], 255)

    if cv2.countNonZero(merged) == 0:
        return None
    return merged


def mask_from_polygon(mask_points, image_shape):
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


def min_area_for_class(class_name: str, args: argparse.Namespace) -> int:
    if class_name == TARGET_PLATE_CLASS:
        return args.target_plate_min_area
    if class_name == "screwdriver_tip":
        return args.screwdriver_tip_min_area
    return 0


def filtered_instances(result, image_shape, args: argparse.Namespace) -> list[dict[str, object]]:
    instances: list[dict[str, object]] = []
    if result.boxes is None:
        return instances

    for index, box in enumerate(result.boxes):
        conf = float(box.conf[0].item())
        if conf < args.conf_thres:
            continue

        cls_id = int(box.cls[0].item())
        class_name = class_name_for_box(result, cls_id)
        mask_points = result_mask_points(result, index)
        binary_mask = mask_from_polygon(mask_points, image_shape)
        if binary_mask is None:
            continue

        area = int(cv2.countNonZero(binary_mask))
        if area < min_area_for_class(class_name, args):
            continue

        instances.append(
            {
                "index": index,
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                "mask_points": mask_points,
                "mask": binary_mask,
                "area": area,
            }
        )

    return instances


def clean_binary_mask(binary_mask, kernel_size: int, open_iterations: int, close_iterations: int):
    if binary_mask is None:
        return None

    import numpy as np

    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return binary_mask
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = binary_mask
    if open_iterations > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    if close_iterations > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    return cleaned


def binary_mask_center(binary_mask, mode: str) -> tuple[int, int] | None:
    import numpy as np

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


def draw_center_marker(image, center_x: int, center_y: int, depth_mm=None) -> None:
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


def instance_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (255, 128, 0),
        (0, 220, 255),
        (80, 180, 80),
        (255, 80, 180),
        (180, 80, 255),
    ]
    return palette[class_id % len(palette)]


def draw_filtered_instances(image, instances: list[dict[str, object]], args: argparse.Namespace):
    annotated = image.copy()
    if args.draw_masks:
        for instance in instances:
            mask = instance["mask"]
            color = instance_color(int(instance["class_id"]))
            colored = annotated.copy()
            colored[mask > 0] = color
            annotated = cv2.addWeighted(colored, 0.35, annotated, 0.65, 0)

    line_width = normalize_line_width(args.line_width)
    if args.draw_boxes or args.draw_labels:
        for instance in instances:
            x1, y1, x2, y2 = [int(round(value)) for value in instance["xyxy"]]
            color = instance_color(int(instance["class_id"]))
            if args.draw_boxes:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)
            if args.draw_labels:
                label = f"{instance['class_name']} {instance['confidence']:.2f}"
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

    return annotated


def draw_detection_centers(
    instances: list[dict[str, object]],
    image,
    depth_image=None,
    mask_center_mode: str = "centroid",
    target_plate_morph_kernel: int = 3,
    target_plate_morph_open: int = 1,
    target_plate_morph_close: int = 1,
) -> list[dict[str, object]]:
    detections: list[dict[str, object]] = []
    if not instances:
        return detections

    target_plate_masks = [
        instance["mask"] for instance in instances if instance["class_name"] == TARGET_PLATE_CLASS
    ]
    target_plate_confs = []
    for instance in instances:
        if instance["class_name"] == TARGET_PLATE_CLASS:
            target_plate_confs.append(float(instance["confidence"]))

    target_plate_exists = bool(target_plate_masks)
    if target_plate_exists:
        import numpy as np

        merged_target_plate_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for binary_mask in target_plate_masks:
            merged_target_plate_mask = cv2.bitwise_or(merged_target_plate_mask, binary_mask)
        merged_target_plate_mask = clean_binary_mask(
            merged_target_plate_mask,
            target_plate_morph_kernel,
            target_plate_morph_open,
            target_plate_morph_close,
        )
        center = (
            None
            if cv2.countNonZero(merged_target_plate_mask) == 0
            else binary_mask_center(merged_target_plate_mask, "bottom")
        )
        if center is not None:
            center_x, center_y = center
            depth_mm = lookup_depth_mm(depth_image, center_x, center_y)
            draw_center_marker(image, center_x, center_y, depth_mm)
            detections.append(
                {
                    "class_id": None,
                    "class_name": TARGET_PLATE_CLASS,
                    "confidence": max(target_plate_confs) if target_plate_confs else 0.0,
                    "center_x": center_x,
                    "center_y": center_y,
                    "center_source": "merged_mask",
                    "depth_mm": depth_mm,
                }
            )

    for instance in instances:
        cls_id = int(instance["class_id"])
        conf = float(instance["confidence"])
        x1, y1, x2, y2 = [float(value) for value in instance["xyxy"]]
        class_name = str(instance["class_name"])
        if class_name == TARGET_PLATE_CLASS:
            continue
        if target_plate_exists and class_name in SUPPRESSED_WHEN_TARGET_EXISTS:
            continue

        binary_mask = instance["mask"]
        center = None if mask_center_mode == "box" else binary_mask_center(binary_mask, mask_center_mode)
        if center is None:
            center_x = int(round((x1 + x2) / 2.0))
            center_y = int(round((y1 + y2) / 2.0))
            center_source = "box"
        else:
            center_x, center_y = center
            center_source = "mask"
        depth_mm = lookup_depth_mm(depth_image, center_x, center_y)

        draw_center_marker(image, center_x, center_y, depth_mm)
        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "center_x": center_x,
                "center_y": center_y,
                "center_source": center_source,
                "depth_mm": depth_mm,
            }
        )

    return detections


def print_detections(frame_id: int, detections: list[dict[str, object]]) -> None:
    if not detections:
        print(f"[FRAME {frame_id}] no detections")
        return

    items = []
    for det in detections:
        depth = det["depth_mm"]
        depth_text = f" z={depth:.0f}mm" if depth is not None else " z=?"
        source_text = f" source={det['center_source']}"
        items.append(
            "{class_name} conf={confidence:.3f} center=({center_x},{center_y})".format(**det)
            + depth_text
            + source_text
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

    def _on_key(self, event) -> None:
        key = event.keysym.lower()
        if key in {"q", "escape"}:
            self.pending_action = "quit"
        elif key == "s":
            self.pending_action = "snapshot"

    def update(self, image) -> str | None:
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


def resolve_window_size(args: argparse.Namespace, image_shape=None) -> tuple[int, int]:
    image_height = args.height
    image_width = args.width
    if image_shape is not None:
        image_height, image_width = image_shape[:2]

    if args.window_width is None:
        width = args.imgsz
    else:
        width = args.window_width

    if args.window_height is None:
        height = int(round(width * image_height / max(1, image_width)))
    else:
        height = args.window_height

    return max(1, int(width)), max(1, int(height))


def create_display(args: argparse.Namespace, image_shape=None):
    title = "YOLO Camera Detect"
    backend = args.display_backend
    window_width, window_height = resolve_window_size(args, image_shape)
    if backend == "opencv":
        return OpenCvDisplay(title)
    if backend == "tkinter":
        return TkinterDisplay(title, window_width, window_height, args.resize_to_window)

    try:
        return OpenCvDisplay(title)
    except cv2.error:
        print("[WARN] OpenCV window is unavailable, falling back to Tkinter display.")
        return TkinterDisplay(title, window_width, window_height, args.resize_to_window)


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
    warned_depth_unaligned = False

    if args.preview_only:
        print("[OK] Preview only: YOLO model is not loaded.")
    else:
        print(f"[OK] Model: {model_path}")
    if args.camera_source == "orbbec":
        print(f"[OK] Camera: Orbbec COLOR_SENSOR via SDK {args.orbbec_sdk_dir}")
        if not args.disable_depth:
            print("[OK] Depth: enabled. Z is read at each detection center.")
    else:
        print(f"[OK] Camera: {args.camera_url if args.camera_url else args.camera_index}")
        print(f"[OK] Backend: {backend_name(resolve_backend(args.backend, is_url=bool(args.camera_url)))}")
    print("[OK] Press q or Esc to quit. Press s to save a snapshot.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("[WARN] Failed to read a frame from camera.")
                break

            if display is None and not args.no_window:
                try:
                    display = create_display(args, frame.shape)
                except (RuntimeError, cv2.error) as exc:
                    print(f"[ERROR] Display window is unavailable: {exc}")
                    print("[HINT] Use --no-window --save-video to run without GUI display.")
                    return

            frame_id += 1
            stats_text = frame_stats_text(frame)
            if args.preview_only:
                annotated = frame.copy()
                detections = []
            else:
                raw_depth = getattr(capture, "last_depth_image", None)
                aligned_to_color = bool(getattr(capture, "depth_aligned_to_color", False))
                depth_for_lookup = prepare_depth_for_lookup(raw_depth, frame.shape, aligned_to_color)
                if raw_depth is not None and depth_for_lookup is None and not warned_depth_unaligned:
                    print(
                        "[WARN] Depth frame is not aligned to the RGB image, so Z lookup is disabled. "
                        "Use --orbbec-depth-align sw or --orbbec-depth-align hw."
                    )
                    warned_depth_unaligned = True

                result = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=max(args.conf, args.conf_thres),
                    iou=args.iou,
                    device=args.device,
                    max_det=args.max_det,
                    verbose=False,
                )[0]
                instances = filtered_instances(result, frame.shape, args)
                annotated = draw_filtered_instances(frame, instances, args)
                detections = draw_detection_centers(
                    instances,
                    annotated,
                    depth_for_lookup,
                    mask_center_mode=args.mask_center_mode,
                    target_plate_morph_kernel=args.target_plate_morph_kernel,
                    target_plate_morph_open=args.target_plate_morph_open,
                    target_plate_morph_close=args.target_plate_morph_close,
                )

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
