#!/usr/bin/env python
"""
Command-line interface for iViewer.

This module sets up the required environment variables (PYOPENGL_PLATFORM=egl)
before importing napari to ensure proper GPU rendering.
"""

import os
import sys
from pathlib import Path

# Set OpenGL platform BEFORE importing napari
# Only set EGL on Red Hat Linux when DISPLAY is not available (headless mode)
# When running in OOD desktop or with X11 forwarding, DISPLAY is set and we should NOT use EGL
if sys.platform == "linux" and Path("/etc/redhat-release").exists():
    if not os.environ.get("DISPLAY"):
        os.environ["PYOPENGL_PLATFORM"] = "egl"

# Note: NAPARI_ASYNC=1 can cause the process to hang on exit due to background threads
# Only enable if needed for performance with large datasets
os.environ["NAPARI_ASYNC"] = "1"


def main():
    """Entry point for the iviewer command."""
    # Import after setting environment variables
    from .viewer import run_viewer
    run_viewer()
    
    # Force immediate exit to terminate any lingering threads
    # os._exit() bypasses cleanup handlers that might be blocking
    os._exit(0)


if __name__ == "__main__":
    main()
