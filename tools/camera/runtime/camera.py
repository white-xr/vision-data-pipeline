from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


BACKENDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "obsensor": getattr(cv2, "CAP_OBSENSOR", cv2.CAP_ANY),
}


@dataclass
class FramePacket:
    color: Any
    depth: Any | None = None
    depth_aligned_to_color: bool = False


def backend_name(api: int) -> str:
    for name, value in BACKENDS.items():
        if value == api:
            return name
    return str(api)


def resolve_backend(name: str, is_url: bool) -> int:
    if is_url:
        return cv2.CAP_ANY if name == "auto" else BACKENDS[name]
    if name == "auto":
        return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    return BACKENDS[name]


def read_frame_with_retries(capture: cv2.VideoCapture, retries: int, delay_seconds: float = 0.1):
    for _ in range(max(1, int(retries))):
        ok, frame = capture.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(delay_seconds)
    return False, None


def maybe_set_opencv_property(capture: cv2.VideoCapture, prop: int, value: Any, name: str) -> None:
    if value is None:
        return
    ok = capture.set(prop, value)
    actual = capture.get(prop)
    if ok:
        print(f"[OK] OpenCV {name}: requested={value}, actual={actual}")
    else:
        print(f"[WARN] OpenCV {name} is not supported by this camera/backend.")


class OpenCvCapture:
    def __init__(self, camera_config: dict[str, Any]) -> None:
        opencv_config = camera_config.get("opencv", {})
        stream_config = camera_config.get("stream", {})
        self.camera_url = opencv_config.get("url")
        self.camera_index = int(opencv_config.get("index", 0))
        source: str | int = self.camera_url if self.camera_url else self.camera_index
        self.api = resolve_backend(str(opencv_config.get("backend", "auto")), is_url=bool(self.camera_url))
        self.capture = cv2.VideoCapture(source, self.api)

        if not self.camera_url:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(stream_config.get("width", 1280)))
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(stream_config.get("height", 720)))
            self.capture.set(cv2.CAP_PROP_FPS, float(stream_config.get("fps", 30.0)))
            self.apply_controls(opencv_config.get("controls", {}))

        if not self.capture.isOpened():
            raise SystemExit(
                f"[ERROR] Cannot open camera source: {source}. "
                "Check camera index, stream URL, permissions, and whether another app is using it."
            )

        ok, _ = read_frame_with_retries(self.capture, int(opencv_config.get("read_retries", 30)))
        if not ok:
            self.capture.release()
            raise SystemExit(
                f"[ERROR] Camera opened but no frame was received: {source}, backend={backend_name(self.api)}.\n"
                "[HINT] Try another index: python tools/camera/camera_detect.py --list-cameras"
            )

    def apply_controls(self, controls: dict[str, Any]) -> None:
        auto_exposure = controls.get("auto_exposure")
        if auto_exposure is not None:
            auto_value = 0.75 if bool(auto_exposure) else 0.25
            maybe_set_opencv_property(self.capture, cv2.CAP_PROP_AUTO_EXPOSURE, auto_value, "auto_exposure")
        maybe_set_opencv_property(self.capture, cv2.CAP_PROP_EXPOSURE, controls.get("exposure"), "exposure")
        maybe_set_opencv_property(self.capture, cv2.CAP_PROP_GAIN, controls.get("gain"), "gain")

    def read(self) -> tuple[bool, FramePacket | None]:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None
        return True, FramePacket(color=frame)

    def get(self, prop: int) -> float:
        return self.capture.get(prop)

    def release(self) -> None:
        self.capture.release()


def add_orbbec_dll_dirs(sdk_dir: Path) -> None:
    bin_dir = sdk_dir / "bin"
    if not bin_dir.is_dir():
        return
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(bin_dir))
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def orbbec_frame_to_bgr(frame: Any, ob_format: Any) -> Any | None:
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


def orbbec_depth_frame_to_mm(depth_frame: Any) -> Any:
    import numpy as np

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16, count=width * height)
    return data.reshape((height, width)).astype("float32") * float(scale)


def set_orbbec_bool_property(device: Any, ob_property_id: Any, name: str, value: Any) -> None:
    if value is None:
        return
    property_id = getattr(ob_property_id, name)
    try:
        device.set_bool_property(property_id, bool(value))
        actual = device.get_bool_property(property_id)
        print(f"[OK] Orbbec {name}: requested={bool(value)}, actual={actual}")
    except Exception as exc:
        print(f"[WARN] Orbbec {name} is not supported or failed to set: {exc}")


def set_orbbec_int_property(device: Any, ob_property_id: Any, name: str, value: Any) -> None:
    if value is None:
        return
    property_id = getattr(ob_property_id, name)
    try:
        device.set_int_property(property_id, int(value))
        actual = device.get_int_property(property_id)
        print(f"[OK] Orbbec {name}: requested={int(value)}, actual={actual}")
    except Exception as exc:
        print(f"[WARN] Orbbec {name} is not supported or failed to set: {exc}")


def apply_orbbec_camera_controls(device: Any, ob_property_id: Any, controls: dict[str, Any]) -> None:
    set_orbbec_bool_property(device, ob_property_id, "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", controls.get("color_auto_exposure"))
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_COLOR_AE_MAX_EXPOSURE_INT", controls.get("color_ae_max_exposure"))
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_COLOR_AE_MAX_GAIN_INT", controls.get("color_ae_max_gain"))
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_COLOR_EXPOSURE_INT", controls.get("color_exposure"))
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_COLOR_GAIN_INT", controls.get("color_gain"))
    set_orbbec_bool_property(device, ob_property_id, "OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL", controls.get("depth_auto_exposure"))
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_DEPTH_EXPOSURE_INT", controls.get("depth_exposure"))
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_DEPTH_GAIN_INT", controls.get("depth_gain"))
    set_orbbec_bool_property(device, ob_property_id, "OB_PROP_IR_AUTO_EXPOSURE_BOOL", controls.get("ir_auto_exposure"))
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_IR_EXPOSURE_INT", controls.get("ir_exposure"))
    set_orbbec_int_property(device, ob_property_id, "OB_PROP_IR_GAIN_INT", controls.get("ir_gain"))


class OrbbecColorCapture:
    def __init__(self, camera_config: dict[str, Any]) -> None:
        self.camera_config = camera_config
        orbbec_config = camera_config.get("orbbec", {})
        stream_config = camera_config.get("stream", {})
        sdk_dir = Path(orbbec_config.get("sdk_dir", "D:/OrbbecSDK_v2"))
        add_orbbec_dll_dirs(sdk_dir)
        try:
            from pyorbbecsdk import (
                Config,
                Context,
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
                f"SDK path checked: {sdk_dir}"
            ) from exc

        self.ob_format = OBFormat
        self.context = None
        serial = orbbec_config.get("serial")
        match_name = orbbec_config.get("match_name")
        if serial or match_name:
            self.context = Context()
            devices = self.context.query_devices()
            if devices.get_count() == 0:
                raise SystemExit("[ERROR] No Orbbec devices were found.")
            if serial:
                device = devices.get_device_by_serial_number(str(serial))
            else:
                match_text = str(match_name).strip().lower()
                matched_indexes = [
                    index
                    for index in range(devices.get_count())
                    if match_text in str(devices.get_device_name_by_index(index)).lower()
                ]
                if not matched_indexes:
                    available = ", ".join(
                        f"{devices.get_device_name_by_index(index)}({devices.get_device_serial_number_by_index(index)})"
                        for index in range(devices.get_count())
                    )
                    raise SystemExit(
                        f"[ERROR] No Orbbec device matched name {match_name!r}. Available: {available}"
                    )
                if len(matched_indexes) > 1:
                    print(
                        f"[WARN] Multiple Orbbec devices matched {match_name!r}; "
                        f"using index {matched_indexes[0]}."
                    )
                device = devices.get_device_by_index(matched_indexes[0])
            self.pipeline = Pipeline(device)
        else:
            self.pipeline = Pipeline()

        self.enable_depth = bool(orbbec_config.get("depth", {}).get("enabled", True))
        self.depth_aligned_to_color = False
        config = Config()
        device = self.pipeline.get_device()
        apply_orbbec_camera_controls(device, OBPropertyID, orbbec_config.get("controls", {}))

        try:
            info = device.get_device_info()
            print(
                "[OK] Orbbec device: "
                f"{info.get_name()}, SN={info.get_serial_number()}, FW={info.get_firmware_version()}"
            )
        except Exception:
            pass

        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = None
        requested_format_name = str(orbbec_config.get("format", "RGB"))
        if requested_format_name != "default":
            requested_format = getattr(OBFormat, requested_format_name)
            try:
                color_profile = profile_list.get_video_stream_profile(
                    int(stream_config.get("width", 1280)),
                    int(stream_config.get("height", 720)),
                    requested_format,
                    int(float(stream_config.get("fps", 30.0))),
                )
            except OBError as exc:
                print(f"[WARN] Requested Orbbec color profile is unavailable: {exc}")

        if color_profile is None:
            color_profile = profile_list.get_default_video_stream_profile()
        print(f"[OK] Orbbec color profile: {color_profile}")
        config.enable_stream(color_profile)

        if self.enable_depth:
            try:
                depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = depth_profiles.get_default_video_stream_profile()
                print(f"[OK] Orbbec depth profile: {depth_profile}")
                config.enable_stream(depth_profile)
                config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

                align_mode = str(orbbec_config.get("depth", {}).get("align", "sw"))
                if align_mode == "hw":
                    config.set_align_mode(OBAlignMode.HW_MODE)
                    self.depth_aligned_to_color = True
                    print("[OK] Orbbec depth align: hardware depth-to-color")
                elif align_mode == "sw":
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
        self.timeout_ms = int(orbbec_config.get("timeout_ms", 1000))
        self.fps = float(stream_config.get("fps", 30.0))

    def read(self) -> tuple[bool, FramePacket | None]:
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return False, None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return False, None

        image = orbbec_frame_to_bgr(color_frame, self.ob_format)
        depth_image = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                depth_image = orbbec_depth_frame_to_mm(depth_frame)

        if image is None:
            return False, None
        return True, FramePacket(
            color=image,
            depth=depth_image,
            depth_aligned_to_color=self.depth_aligned_to_color,
        )

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def release(self) -> None:
        self.pipeline.stop()


def open_camera(camera_config: dict[str, Any]):
    source = str(camera_config.get("source", "opencv")).lower()
    if source == "orbbec":
        return OrbbecColorCapture(camera_config)
    if source == "opencv":
        return OpenCvCapture(camera_config)
    raise SystemExit(f"[ERROR] Unknown camera source: {source}")


def frame_stats_text(image: Any) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    min_value, max_value, _, _ = cv2.minMaxLoc(gray)
    mean_value = float(gray.mean())
    return f"min={min_value:.0f} max={max_value:.0f} mean={mean_value:.1f}"


def list_opencv_cameras(max_index: int = 10) -> None:
    print("[INFO] Scanning OpenCV camera indexes...")
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
                    f"[OK] OpenCV index={index}, backend={backend}, "
                    f"frame={actual_width}x{actual_height}, requested={width}x{height}, "
                    f"fps={fps:.1f}, {frame_stats_text(frame)}"
                )
                found = True
            else:
                print(f"[WARN] OpenCV index={index}, backend={backend}, opened but no frame")
    if not found:
        print("[WARN] No readable OpenCV camera was found.")


def list_orbbec_devices(camera_config: dict[str, Any]) -> None:
    print("[INFO] Scanning Orbbec devices...")
    sdk_dir = Path(camera_config.get("orbbec", {}).get("sdk_dir", "D:/OrbbecSDK_v2"))
    add_orbbec_dll_dirs(sdk_dir)
    try:
        from pyorbbecsdk import Context
    except ImportError as exc:
        print(f"[WARN] Cannot import pyorbbecsdk: {exc}")
        return

    context = Context()
    devices = context.query_devices()
    count = devices.get_count()
    if count == 0:
        print("[WARN] No Orbbec devices were found.")
        return
    for index in range(count):
        print(
            "[OK] Orbbec "
            f"index={index}, name={devices.get_device_name_by_index(index)}, "
            f"serial={devices.get_device_serial_number_by_index(index)}, "
            f"uid={devices.get_device_uid_by_index(index)}, "
            f"connection={devices.get_device_connection_type_by_index(index)}"
        )


def list_cameras(camera_config: dict[str, Any]) -> None:
    list_orbbec_devices(camera_config)
    list_opencv_cameras()
