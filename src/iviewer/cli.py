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
# Only set EGL on Red Hat Linux (required for GPU rendering on HPC clusters)
if sys.platform == "linux" and Path("/etc/redhat-release").exists():
    os.environ["PYOPENGL_PLATFORM"] = "egl"

os.environ["NAPARI_ASYNC"] = "1"


def main():
    """Entry point for the iviewer command."""
    # Import after setting environment variables
    from .viewer import run_viewer
    run_viewer()


if __name__ == "__main__":
    main()
