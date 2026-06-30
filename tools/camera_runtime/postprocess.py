from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable


PostprocessCallable = Callable[[list[dict[str, Any]], Any, Any, dict[str, Any]], Any]


@dataclass
class PostProcessor:
    enabled: bool
    func: PostprocessCallable | None = None
    params: dict[str, Any] | None = None
    label: str = "disabled"

    def process(self, detections: list[dict[str, Any]], frame: Any, depth_image: Any) -> dict[str, Any]:
        if not self.enabled or self.func is None:
            return {"detections": detections, "status_lines": []}
        result = self.func(detections, frame, depth_image, self.params or {})
        if result is None:
            return {"detections": detections, "status_lines": []}
        if isinstance(result, list):
            return {"detections": result, "status_lines": []}
        if not isinstance(result, dict):
            raise RuntimeError(f"Postprocess plugin {self.label} returned {type(result).__name__}, expected dict.")
        return {
            "detections": result.get("detections", detections),
            "status_lines": list(result.get("status_lines", [])),
        }


def load_post_processor(config: dict[str, Any]) -> PostProcessor:
    if not bool(config.get("enabled", False)):
        return PostProcessor(enabled=False)
    module_name = config.get("module")
    if not module_name:
        raise SystemExit("[ERROR] postprocess.enabled is true but postprocess.module is empty.")
    function_name = str(config.get("function", "process"))
    try:
        module = importlib.import_module(str(module_name))
    except ImportError as exc:
        raise SystemExit(f"[ERROR] Cannot import postprocess module {module_name!r}: {exc}") from exc
    func = getattr(module, function_name, None)
    if func is None or not callable(func):
        raise SystemExit(f"[ERROR] Postprocess function not found: {module_name}.{function_name}")
    return PostProcessor(
        enabled=True,
        func=func,
        params=dict(config.get("params", {}) or {}),
        label=f"{module_name}.{function_name}",
    )
