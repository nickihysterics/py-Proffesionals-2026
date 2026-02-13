"""Точка входа для `python -m gaze_tracker`."""

from __future__ import annotations

import sys

from gaze_tracker.api import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
