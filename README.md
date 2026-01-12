# iViewer

iFISH 3D Visualization with napari.

Interactive exploration of spatial transcriptomic data:
- Reference DAPI and GFP channels
- Cell segmentation with 3D bounding boxes
- Registered puncta images from multiple cycles
- Detected puncta centroid locations

## Installation

```bash
# Create and activate a virtual environment
python -m venv /path/to/venv/iviewer
source /path/to/venv/iviewer/bin/activate

# Install in editable mode (for development)
pip install -e /scratch/napari_viewer_0109/iviewer

# Or install directly
pip install /scratch/napari_viewer_0109/iviewer
```

## Usage

The `iviewer` command automatically sets `PYOPENGL_PLATFORM=egl` for GPU rendering.

### View a reference image only

```bash
iviewer --ref /path/to/reference_image.tif
```

### View full cycleHCR data directory

```bash
iviewer --dir /path/to/data_directory
```

The data directory should contain:
- `segmentation/`: Reference images and cell masks
- `regis_puncta_img/`: Registered puncta images
- `regis_puncta_loc/pixel/`: Puncta location CSVs

### Enable orthogonal views

```bash
iviewer --ref /path/to/image.tif --ortho
```

## Examples

```bash
# Simple reference image viewing
iviewer --ref /scratch/Gel_1029/segmentation_1212/test_norm_background_1216/data/transformations/gamma_0.5/Gel20251024_round00_brain08_intact_cropped.tif

# Full data directory
iviewer --dir /path/to/cycleHCR_data

# With orthogonal views
iviewer --ref /path/to/image.tif --ortho
```

## Python API

You can also use iviewer as a library:

```python
from iviewer import load_reference_image, CHANNEL_COLORS

# Load and split a reference image
dapi, gfp, ref_z = load_reference_image("/path/to/image.tif")
```
