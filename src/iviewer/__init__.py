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

from .reader import get_reader, read_puncta_csv, read_mask, read_h5, read_nd2

from .group_widget import LayerGroupsWidget
from .mask_highlight_widget import MaskHighlightWidget
from .mask_editor_widget import MaskEditorWidget

__all__ = [
    "__version__",
    "load_reference_image",
    "load_reference_and_masks",
    "add_puncta_images",
    "add_puncta_locations",
    "CHANNEL_COLORS",
    "get_reader",
    "read_puncta_csv",
    "read_mask",
    "read_h5",
    "read_nd2",
    "LayerGroupsWidget",
    "MaskHighlightWidget",
    "MaskEditorWidget",
]
