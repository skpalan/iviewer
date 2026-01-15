#!/usr/bin/env python
"""
Custom napari readers for 3D TIFF images and puncta CSV files.

This module provides napari reader plugins that properly handle:
- 3D TIFF images with correct dimension ordering
- Puncta location CSV files as points layers

Common dimension orderings for TIFF:
- (Z, Y, X): Standard 3D volume
- (Z, C, Y, X): 3D multichannel
- (C, Z, Y, X): Alternative multichannel
- (T, Y, X): Time series
"""

from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import imageio.v3 as iio
import nd2
import numpy as np
import pandas as pd
from tifffile import TiffFile, imread as tiff_imread

# Channel color mapping for microscopy channels
CHANNEL_COLORS = {
    "Cy3": "yellow",
    "Cy5": "red",
    "mCherry": "magenta",
    "DAPI": "blue",
    "GFP": "green",
    "FITC": "green",
    "FITC-GFP": "green",
}


def get_reader(path: Union[str, List[str]]) -> Optional[Callable]:
    """Return reader function if path is a supported file (TIFF, PNG, H5, ND2, or CSV).
    
    Parameters
    ----------
    path : str or list of str
        Path(s) to file(s).
        
    Returns
    -------
    callable or None
        Reader function if path is supported, None otherwise.
    """
    if isinstance(path, list):
        # Handle multiple files - check if all are same type
        if all(_is_mask_file(p) for p in path):
            return read_mask
        if all(_is_nd2_file(p) for p in path):
            return read_nd2
        if all(_is_h5_file(p) for p in path):
            return read_h5
        if all(_is_supported_image(p) for p in path):
            return read_tiff_stack
        if all(_is_puncta_csv(p) for p in path):
            return read_puncta_csv
        return None
    
    # Check mask files first (more specific than general image)
    if _is_mask_file(path):
        return read_mask
    
    if _is_nd2_file(path):
        return read_nd2
    
    if _is_h5_file(path):
        return read_h5
    
    if _is_supported_image(path):
        return read_tiff_stack
    
    if _is_puncta_csv(path):
        return read_puncta_csv
    
    return None


def _is_supported_image(path: str) -> bool:
    """Check if path is a supported image file (TIFF or PNG)."""
    path = Path(path)
    return path.suffix.lower() in {'.tif', '.tiff', '.png'}


def _is_mask_file(path: str) -> bool:
    """Check if path is a segmentation mask file.
    
    A file is considered a mask if:
    - It's a TIFF or PNG file
    - Its filename contains 'mask' (case-insensitive)
    """
    path = Path(path)
    if path.suffix.lower() not in {'.tif', '.tiff', '.png'}:
        return False
    return 'mask' in path.stem.lower()


def _is_h5_file(path: str) -> bool:
    """Check if path is an HDF5 file."""
    path = Path(path)
    return path.suffix.lower() in {'.h5', '.hdf5'}


def _is_nd2_file(path: str) -> bool:
    """Check if path is a Nikon ND2 file."""
    path = Path(path)
    return path.suffix.lower() == '.nd2'


def _is_puncta_csv(path: str) -> bool:
    """Check if path is a puncta location CSV file."""
    path = Path(path)
    if path.suffix.lower() != '.csv':
        return False
    
    # Quick check: read first line to verify it has expected columns
    try:
        with open(path, 'r') as f:
            header = f.readline().strip().lower()
            # Must have x, y, z columns for spatial data
            return 'x' in header and 'y' in header and 'z' in header
    except Exception:
        return False


# Default channel names for common microscopy setups
DEFAULT_CHANNEL_NAMES = ['DAPI', 'GFP', 'Cy3', 'Cy5', 'mCherry', 'BFP', 'YFP', 'RFP']


def _get_channel_names(n_channels: int) -> List[str]:
    """Get channel names for a multichannel image.
    
    Parameters
    ----------
    n_channels : int
        Number of channels in the image.
        
    Returns
    -------
    list of str
        List of channel names.
    """
    if n_channels <= len(DEFAULT_CHANNEL_NAMES):
        return DEFAULT_CHANNEL_NAMES[:n_channels]
    else:
        # For more channels than defaults, use Ch1, Ch2, etc. for extras
        names = DEFAULT_CHANNEL_NAMES.copy()
        for i in range(len(DEFAULT_CHANNEL_NAMES), n_channels):
            names.append(f'Ch{i + 1}')
        return names


def _detect_dimension_order(tif: TiffFile, data: np.ndarray) -> Tuple[str, dict]:
    """Detect dimension order from TIFF metadata and data shape.
    
    Parameters
    ----------
    tif : TiffFile
        Open TiffFile object with metadata.
    data : np.ndarray
        Image data array.
        
    Returns
    -------
    tuple of (str, dict)
        Tuple of (axis_labels, metadata) where axis_labels is a string like 'ZYX'
        and metadata contains information about the detection.
    """
    shape = data.shape
    ndim = data.ndim
    metadata = {'original_shape': shape, 'detection_method': None}
    
    # Try to get axis information from tifffile
    # tifffile stores axis info in series[0].axes
    axes = None
    if tif.series:
        axes = tif.series[0].axes
        if axes:
            metadata['tiff_axes'] = axes
            metadata['detection_method'] = 'tiff_metadata'
            return axes, metadata
    
    # Check ImageJ metadata
    if tif.imagej_metadata:
        ij_meta = tif.imagej_metadata
        metadata['imagej_metadata'] = ij_meta
        
        # ImageJ stores dimension info
        n_slices = ij_meta.get('slices', 1)
        n_channels = ij_meta.get('channels', 1)
        n_frames = ij_meta.get('frames', 1)
        
        if n_slices > 1 or n_channels > 1 or n_frames > 1:
            # Build axis string based on ImageJ metadata
            axes_parts = []
            if n_frames > 1:
                axes_parts.append('T')
            if n_channels > 1:
                axes_parts.append('C')
            if n_slices > 1:
                axes_parts.append('Z')
            axes_parts.extend(['Y', 'X'])
            
            axes = ''.join(axes_parts)
            metadata['detection_method'] = 'imagej_metadata'
            return axes, metadata
    
    # Heuristic detection based on shape
    metadata['detection_method'] = 'heuristic'
    
    if ndim == 2:
        return 'YX', metadata
    
    elif ndim == 3:
        # For 3D images, determine if first dim is Z, C, or T
        # Heuristic: Z typically has more slices than channels
        # Channels are usually small (1-4), Z can be large (10-1000+)
        first_dim = shape[0]
        
        # If first dimension is small (<=4), might be channels
        # But for 3D TIFF, assume ZYX by default (most common for microscopy)
        if first_dim <= 4 and shape[1] > first_dim and shape[2] > first_dim:
            # Could be CYX, but we'll still assume ZYX and let user adjust
            # For spatial data, ZYX is more common
            pass
        
        # Default assumption: ZYX (most common for 3D microscopy data)
        return 'ZYX', metadata
    
    elif ndim == 4:
        # 4D: could be ZCYX, CZYX, TZYX, TCYX
        first_dim = shape[0]
        second_dim = shape[1]
        
        # Heuristic: if second dim is small (<=4), likely ZCYX
        # If first dim is small (<=4), likely CZYX
        if second_dim <= 4:
            return 'ZCYX', metadata
        elif first_dim <= 4:
            return 'CZYX', metadata
        else:
            # Default to TZYX for large first two dims
            return 'TZYX', metadata
    
    elif ndim == 5:
        # 5D: TZCYX or TCZYX
        if shape[2] <= 4:
            return 'TZCYX', metadata
        else:
            return 'TCZYX', metadata
    
    # Fallback: just use dimension indices
    return ''.join([f'D{i}' for i in range(ndim - 2)] + ['Y', 'X']), metadata


def _reorder_to_zyx(data: np.ndarray, axes: str) -> Tuple[np.ndarray, dict]:
    """Reorder data to ensure ZYX order for 3D volumes.
    
    Parameters
    ----------
    data : np.ndarray
        Image data.
    axes : str
        Current axis labels (e.g., 'ZYX', 'XYZ', 'ZCY X').
        
    Returns
    -------
    tuple of (np.ndarray, dict)
        Reordered data and layer kwargs for napari.
    """
    axes = axes.upper().replace(' ', '')
    shape = data.shape
    layer_kwargs = {}
    
    # Handle common reorderings
    if data.ndim == 3:
        if axes == 'ZYX':
            # Already correct
            return data, layer_kwargs
        elif axes == 'YXZ':
            # Transpose to ZYX
            return np.transpose(data, (2, 0, 1)), layer_kwargs
        elif axes == 'XYZ':
            # Transpose to ZYX
            return np.transpose(data, (2, 1, 0)), layer_kwargs
        elif axes == 'ZXY':
            # Transpose to ZYX
            return np.transpose(data, (0, 2, 1)), layer_kwargs
        elif axes == 'CYX':
            # Multichannel 2D - let napari handle as-is
            layer_kwargs['channel_axis'] = 0
            return data, layer_kwargs
    
    elif data.ndim == 4:
        if axes == 'ZCYX':
            # Split channels
            layer_kwargs['channel_axis'] = 1
            return data, layer_kwargs
        elif axes == 'CZYX':
            # Channel first - reorder to ZCYX
            data = np.transpose(data, (1, 0, 2, 3))
            layer_kwargs['channel_axis'] = 1
            return data, layer_kwargs
        elif axes == 'ZYXC':
            # Channel last - reorder to ZCYX
            data = np.transpose(data, (0, 3, 1, 2))
            layer_kwargs['channel_axis'] = 1
            return data, layer_kwargs
        elif axes == 'TZYX':
            # Time series - no channel axis
            return data, layer_kwargs
    
    # Default: return as-is
    return data, layer_kwargs


def read_tiff_stack(path: Union[str, List[str]]) -> List[Tuple]:
    """Read TIFF file(s) and return napari layer data.
    
    Parameters
    ----------
    path : str or list of str
        Path(s) to TIFF file(s).
        
    Returns
    -------
    list of tuple
        List of (data, kwargs, layer_type) tuples for napari.
    """
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path
    
    layers = []
    
    for file_path in paths:
        file_path = Path(file_path)
        name = file_path.stem
        
        # Read with tifffile to get metadata
        with TiffFile(file_path) as tif:
            data = tif.asarray()
            axes, detection_meta = _detect_dimension_order(tif, data)
        
        # Print detection info for debugging
        print(f"Loading: {name}")
        print(f"  Shape: {data.shape}")
        print(f"  Detected axes: {axes}")
        print(f"  Detection method: {detection_meta.get('detection_method', 'unknown')}")
        
        # Reorder data if needed
        data, layer_kwargs = _reorder_to_zyx(data, axes)
        
        if data.shape != detection_meta['original_shape']:
            print(f"  Reordered shape: {data.shape}")
        
        # Set layer name(s) - format: {channel_name}-{filename}
        if 'channel_axis' in layer_kwargs:
            # Multichannel image - provide names for each channel
            n_channels = data.shape[layer_kwargs['channel_axis']]
            channel_names = _get_channel_names(n_channels)
            layer_kwargs['name'] = [f"{ch}-{name}" for ch in channel_names]
            print(f"  Channels: {channel_names}")
        else:
            layer_kwargs['name'] = name
        
        # Add blending for better visualization
        layer_kwargs['blending'] = 'additive'
        
        layers.append((data, layer_kwargs, 'image'))
    
    return layers


def _parse_puncta_filename(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse round and channel info from puncta CSV filename.
    
    Expected filename patterns:
    - Thr_0p00174_Gel20251024_round08_brain08_channel-Cy3 Nar_regis
    - {prefix}_{round_id}_{...}_channel-{channel}_...
    
    Parameters
    ----------
    name : str
        Filename stem (without extension).
        
    Returns
    -------
    tuple of (str or None, str or None)
        (round_id, channel) extracted from filename, or (None, None) if parsing fails.
    """
    parts = name.split("_")
    round_id = None
    channel = None
    
    # Find round ID (e.g., "round08")
    for part in parts:
        if part.startswith("round"):
            round_id = part
            break
    
    # Find channel (after "channel-")
    if "channel-" in name:
        try:
            channel = name.split("channel-")[1].split("_")[0]
        except (IndexError, ValueError):
            pass
    
    return round_id, channel


def read_puncta_csv(path: Union[str, List[str]]) -> List[Tuple]:
    """Read puncta location CSV file(s) and return napari points layer data.
    
    Parameters
    ----------
    path : str or list of str
        Path(s) to CSV file(s).
        
    Returns
    -------
    list of tuple
        List of (data, kwargs, layer_type) tuples for napari.
    """
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path
    
    layers = []
    
    for file_path in paths:
        file_path = Path(file_path)
        name = file_path.stem
        
        # Read CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {name}: {e}")
            continue
        
        if len(df) == 0:
            print(f"Skipping {name}: empty file")
            continue
        
        # Check for required columns
        required_cols = ['x', 'y', 'z']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Skipping {name}: missing columns {missing_cols}")
            continue
        
        # Extract coordinates (napari expects z, y, x order)
        coords = df[["z", "y", "x"]].values.copy()
        
        # Parse filename to get round and channel info
        round_id, channel = _parse_puncta_filename(name)
        
        # Determine color based on channel
        if channel:
            channel_base = channel.replace(" Nar", "")
            color = CHANNEL_COLORS.get(channel_base, "white")
        else:
            color = "white"
        
        # Build layer name
        if round_id and channel:
            layer_name = f"Loc {round_id} {channel}"
        else:
            layer_name = f"Loc {name}"
        
        # Print loading info
        print(f"Loading: {name}")
        print(f"  Points: {len(df)}")
        if round_id:
            print(f"  Round: {round_id}")
        if channel:
            print(f"  Channel: {channel} (color: {color})")
        
        # Use arrays for size and face_color to enable interactive editing in napari GUI
        n_points = len(coords)
        # Convert color name to RGBA array for per-point color editing
        from napari.utils.colormaps.standardize_color import transform_color
        rgba = transform_color(color)[0]  # Get RGBA values
        face_colors = np.tile(rgba, (n_points, 1))  # (N, 4) array
        
        layer_kwargs = {
            'name': layer_name,
            'face_color': face_colors,
            'size': np.full(n_points, 3, dtype=np.float32),
            'blending': 'translucent',
        }
        
        layers.append((coords, layer_kwargs, 'points'))
    
    return layers


def read_mask(path: Union[str, List[str]]) -> List[Tuple]:
    """Read segmentation mask file(s) and return napari labels layer data.
    
    Parameters
    ----------
    path : str or list of str
        Path(s) to mask file(s) (TIFF or PNG with 'mask' in filename).
        
    Returns
    -------
    list of tuple
        List of (data, kwargs, layer_type) tuples for napari.
    """
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path
    
    layers = []
    
    for file_path in paths:
        file_path = Path(file_path)
        name = file_path.stem
        suffix = file_path.suffix.lower()
        
        # Read the mask file
        try:
            if suffix in {'.tif', '.tiff'}:
                with TiffFile(file_path) as tif:
                    data = tif.asarray()
            else:  # PNG
                data = iio.imread(file_path)
        except Exception as e:
            print(f"Error reading mask {name}: {e}")
            continue
        
        # Ensure integer type for labels
        if not np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.int32)
        
        # Count unique labels (excluding background 0)
        n_labels = len(np.unique(data)) - 1 if 0 in data else len(np.unique(data))
        
        print(f"Loading mask: {name}")
        print(f"  Shape: {data.shape}")
        print(f"  Labels: {n_labels}")
        
        layer_kwargs = {
            'name': name,
            'opacity': 0.5,
        }
        
        layers.append((data, layer_kwargs, 'labels'))
    
    return layers


def _parse_h5_filename(name: str) -> Optional[str]:
    """Parse channel info from H5 filename.
    
    Expected filename pattern:
    - Gel20260109_round01_brain01_channel-DAPI
    
    Parameters
    ----------
    name : str
        Filename stem (without extension).
        
    Returns
    -------
    str or None
        Channel name extracted from filename, or None if parsing fails.
    """
    if "channel-" in name:
        try:
            return name.split("channel-")[1].split("_")[0]
        except (IndexError, ValueError):
            pass
    return None


def read_h5(path: Union[str, List[str]]) -> List[Tuple]:
    """Read HDF5 file(s) and return napari image layer data.
    
    Expects H5 files with a 'data' dataset containing the image array.
    
    Parameters
    ----------
    path : str or list of str
        Path(s) to H5 file(s).
        
    Returns
    -------
    list of tuple
        List of (data, kwargs, layer_type) tuples for napari.
    """
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path
    
    layers = []
    
    for file_path in paths:
        file_path = Path(file_path)
        name = file_path.stem
        
        # Read the H5 file
        try:
            with h5py.File(file_path, 'r') as f:
                # Try common dataset names
                if 'data' in f:
                    data = f['data'][:]
                elif 'image' in f:
                    data = f['image'][:]
                elif 'volume' in f:
                    data = f['volume'][:]
                else:
                    # Use first dataset found
                    keys = list(f.keys())
                    if keys:
                        data = f[keys[0]][:]
                    else:
                        print(f"Skipping {name}: no datasets found in H5 file")
                        continue
        except Exception as e:
            print(f"Error reading H5 file {name}: {e}")
            continue
        
        # Parse channel from filename for coloring
        channel = _parse_h5_filename(name)
        if channel:
            channel_base = channel.replace(" Nar", "")
            color = CHANNEL_COLORS.get(channel_base, "gray")
        else:
            color = "gray"
        
        print(f"Loading H5: {name}")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        if channel:
            print(f"  Channel: {channel} (color: {color})")
        
        layer_kwargs = {
            'name': name,
            'colormap': color,
            'blending': 'additive',
        }
        
        layers.append((data, layer_kwargs, 'image'))
    
    return layers


def read_nd2(path: Union[str, List[str]]) -> List[Tuple]:
    """Read Nikon ND2 file(s) and return napari image layer data.
    
    Handles duplicate channels by averaging them. For example, if there are
    three Cy3 channels, they will be averaged into a single Cy3 layer.
    
    Parameters
    ----------
    path : str or list of str
        Path(s) to ND2 file(s).
        
    Returns
    -------
    list of tuple
        List of (data, kwargs, layer_type) tuples for napari.
    """
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path
    
    layers = []
    
    for file_path in paths:
        file_path = Path(file_path)
        name = file_path.stem
        
        try:
            with nd2.ND2File(file_path) as f:
                # Get full data array - shape is typically (Z, C, Y, X)
                data = f.asarray()
                
                # Get channel names from metadata
                channel_names = []
                if f.metadata and f.metadata.channels:
                    channel_names = [ch.channel.name for ch in f.metadata.channels]
                
                print(f"Loading ND2: {name}")
                print(f"  Shape: {data.shape}")
                print(f"  Channels: {channel_names}")
                
                # Group channels by name to handle duplicates
                # data shape is (Z, C, Y, X)
                if len(channel_names) == data.shape[1]:
                    # Group channel indices by name
                    channel_groups: Dict[str, List[int]] = defaultdict(list)
                    for idx, ch_name in enumerate(channel_names):
                        channel_groups[ch_name].append(idx)
                    
                    print(f"  Channel groups: {dict(channel_groups)}")
                    
                    # Process each unique channel
                    for ch_name, indices in channel_groups.items():
                        if len(indices) > 1:
                            # Average duplicate channels
                            ch_data = data[:, indices, :, :].mean(axis=1)
                            print(f"  Averaged {len(indices)} '{ch_name}' channels")
                        else:
                            ch_data = data[:, indices[0], :, :]
                        
                        # Get color for this channel
                        color = CHANNEL_COLORS.get(ch_name, "gray")
                        
                        layer_kwargs = {
                            'name': f"{ch_name}-{name}",
                            'colormap': color,
                            'blending': 'additive',
                        }
                        
                        layers.append((ch_data, layer_kwargs, 'image'))
                else:
                    # Fallback: no channel metadata or mismatch, load as-is
                    print(f"  Warning: channel count mismatch, loading raw data")
                    layer_kwargs = {
                        'name': name,
                        'blending': 'additive',
                    }
                    # If 4D with channel axis, set channel_axis
                    if data.ndim == 4:
                        layer_kwargs['channel_axis'] = 1
                    layers.append((data, layer_kwargs, 'image'))
                    
        except Exception as e:
            print(f"Error reading ND2 file {name}: {e}")
            continue
    
    return layers
