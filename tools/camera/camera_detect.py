"""Start the generic camera inference runtime.

Examples:
    python tools/camera/camera_detect.py
    python tools/camera/camera_detect.py --dry-run
"""

from __future__ import annotations

import sys

from runtime.app import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
