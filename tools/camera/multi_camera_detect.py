"""Start named camera pipelines with serial YOLO models.

Examples:
    python tools/camera/multi_camera_detect.py --dry-run
    python tools/camera/multi_camera_detect.py --pipeline base_cover_335l
"""

from __future__ import annotations

import sys

from runtime.multi_app import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
