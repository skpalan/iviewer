"""
iViewer - cycleHCR 3D Visualization with napari

Interactive exploration of spatial transcriptomic data:
- Reference DAPI and GFP channels
- Cell segmentation with 3D bounding boxes
- Registered puncta images from multiple cycles
- Detected puncta centroid locations
"""

__version__ = "0.1.0"

from .viewer import (
    load_reference_image,
    load_reference_and_masks,
    add_puncta_images,
    add_puncta_locations,
    CHANNEL_COLORS,
)

__all__ = [
    "__version__",
    "load_reference_image",
    "load_reference_and_masks",
    "add_puncta_images",
    "add_puncta_locations",
    "CHANNEL_COLORS",
]
