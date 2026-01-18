"""
RS-FISH Widget for napari

Provides interactive RS-FISH spot detection by wrapping the RS-FISH CLI tool.
Allows parameter tuning with preview functionality and 3D visualization of detected spots.

RS-FISH: Precise, interactive, fast, and scalable FISH spot detection
https://github.com/PreibischLab/RS-FISH
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSlider,
    QPushButton,
    QFrame,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QProgressBar,
    QMessageBox,
    QTextEdit,
)
from tifffile import imwrite

import napari
from napari.layers import Image, Points, Shapes


def find_rsfish_cli() -> Optional[str]:
    """Find the rs-fish CLI executable.
    
    Returns the path to rs-fish if found, None otherwise.
    """
    # Check if rs-fish is on PATH
    rsfish_path = shutil.which("rs-fish")
    if rsfish_path:
        return rsfish_path
    
    # Check common installation locations
    common_paths = [
        Path.home() / "bin" / "rs-fish",
        Path.home() / ".local" / "bin" / "rs-fish",
        Path("/usr/local/bin/rs-fish"),
    ]
    
    for path in common_paths:
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    
    return None


class RSFISHWorker(QThread):
    """Worker thread for running RS-FISH detection."""
    
    finished = Signal(object)  # Emits DataFrame or None on error
    error = Signal(str)  # Emits error message
    progress = Signal(str)  # Emits progress messages
    
    def __init__(
        self,
        rsfish_path: str,
        image_path: str,
        output_path: str,
        params: dict,
    ):
        super().__init__()
        self.rsfish_path = rsfish_path
        self.image_path = image_path
        self.output_path = output_path
        self.params = params
    
    def run(self):
        """Execute RS-FISH CLI."""
        try:
            # Build command
            # Note: CLI parameter names differ from Fiji macro names:
            # CLI uses: --supportRadius, --inlierRatio, --maxError
            # Fiji macro uses: support, min_inlier_ratio, max_error
            cmd = [
                self.rsfish_path,
                f"--image={self.image_path}",
                f"--output={self.output_path}",
                f"--sigma={self.params['sigma']}",
                f"--threshold={self.params['threshold']}",
                f"--anisotropy={self.params['anisotropy']}",
                f"--supportRadius={self.params['support']}",
                f"--inlierRatio={self.params['inlier_ratio']}",
                f"--maxError={self.params['max_error']}",
                f"--background={self.params['background']}",
            ]
            
            # Add optional intensity range
            if not self.params.get('auto_minmax', True):
                cmd.append(f"--minIntensity={self.params['min_intensity']}")
                cmd.append(f"--maxIntensity={self.params['max_intensity']}")
            
            self.progress.emit(f"Running: {' '.join(cmd)}")
            
            # Run RS-FISH
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            
            if result.returncode != 0:
                self.error.emit(f"RS-FISH failed:\n{result.stderr}")
                return
            
            self.progress.emit(result.stdout)
            
            # Load results
            if Path(self.output_path).exists():
                df = pd.read_csv(self.output_path)
                self.finished.emit(df)
            else:
                self.error.emit("RS-FISH completed but no output file was created")
                
        except subprocess.TimeoutExpired:
            self.error.emit("RS-FISH timed out after 10 minutes")
        except Exception as e:
            self.error.emit(f"Error running RS-FISH: {str(e)}")


class RSFISHWidget(QWidget):
    """Widget for interactive RS-FISH spot detection in napari.
    
    This widget wraps the RS-FISH command-line tool to provide:
    - Interactive parameter tuning
    - Preview mode for quick parameter adjustment
    - Full 3D detection
    - Results visualization as napari Points layer
    
    Prerequisites:
        RS-FISH CLI must be installed. See: https://github.com/PreibischLab/RS-FISH
    
    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer instance.
    """
    
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # Check for RS-FISH CLI
        self.rsfish_path = find_rsfish_cli()
        
        # Temporary directory for intermediate files
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        
        # Worker thread for background processing
        self._worker: Optional[RSFISHWorker] = None
        
        # Track created layers
        self._points_layer_name = "RS-FISH Detections"
        self._preview_layer_name = "RS-FISH Preview"
        self._roi_layer_name = "RS-FISH ROI"
        
        self._setup_ui()
        self._connect_signals()
        self._refresh_layer_list()
        
        # Show warning if RS-FISH not found
        if not self.rsfish_path:
            self._show_rsfish_not_found()
    
    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header
        header = QLabel("<b>RS-FISH Spot Detection</b>")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # CLI status indicator
        self.lbl_cli_status = QLabel()
        self.lbl_cli_status.setAlignment(Qt.AlignCenter)
        self._update_cli_status()
        layout.addWidget(self.lbl_cli_status)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Image selection
        img_group = QGroupBox("Image Selection")
        img_layout = QVBoxLayout(img_group)
        
        self.combo_image = QComboBox()
        self.combo_image.setToolTip("Select the image layer to detect spots in")
        img_layout.addWidget(self.combo_image)
        
        self.chk_auto_minmax = QCheckBox("Compute min/max from image")
        self.chk_auto_minmax.setChecked(True)
        self.chk_auto_minmax.setToolTip("Automatically compute intensity range from image")
        img_layout.addWidget(self.chk_auto_minmax)
        
        layout.addWidget(img_group)
        
        # DoG Parameters
        dog_group = QGroupBox("DoG Detection")
        dog_layout = QVBoxLayout(dog_group)
        
        # Sigma
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma:"))
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.5, 10.0)
        self.spin_sigma.setValue(1.5)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setToolTip("Sigma for Difference of Gaussian (spot size)")
        sigma_layout.addWidget(self.spin_sigma)
        dog_layout.addLayout(sigma_layout)
        
        # Threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold:"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 10000.0)
        self.spin_threshold.setValue(0.007)
        self.spin_threshold.setDecimals(6)
        self.spin_threshold.setSingleStep(0.001)
        self.spin_threshold.setToolTip("Detection threshold (higher = fewer spots)")
        thresh_layout.addWidget(self.spin_threshold)
        dog_layout.addLayout(thresh_layout)
        
        layout.addWidget(dog_group)
        
        # Anisotropy
        aniso_group = QGroupBox("Anisotropy")
        aniso_layout = QVBoxLayout(aniso_group)
        
        aniso_val_layout = QHBoxLayout()
        aniso_val_layout.addWidget(QLabel("Coefficient:"))
        self.spin_anisotropy = QDoubleSpinBox()
        self.spin_anisotropy.setRange(0.1, 10.0)
        self.spin_anisotropy.setValue(1.0)
        self.spin_anisotropy.setSingleStep(0.1)
        self.spin_anisotropy.setToolTip("Z-axis correction factor (z_size / xy_size)")
        aniso_val_layout.addWidget(self.spin_anisotropy)
        aniso_layout.addLayout(aniso_val_layout)
        
        self.btn_calc_anisotropy = QPushButton("Calculate from image...")
        self.btn_calc_anisotropy.setToolTip("Calculate anisotropy coefficient using rs-fish-anisotropy")
        self.btn_calc_anisotropy.clicked.connect(self._calculate_anisotropy)
        aniso_layout.addWidget(self.btn_calc_anisotropy)
        
        layout.addWidget(aniso_group)
        
        # RANSAC Parameters
        ransac_group = QGroupBox("RANSAC Parameters")
        ransac_layout = QVBoxLayout(ransac_group)
        
        # Support region
        support_layout = QHBoxLayout()
        support_layout.addWidget(QLabel("Support radius:"))
        self.spin_support = QSpinBox()
        self.spin_support.setRange(1, 10)
        self.spin_support.setValue(3)
        self.spin_support.setToolTip("Support region radius in pixels")
        support_layout.addWidget(self.spin_support)
        ransac_layout.addLayout(support_layout)
        
        # Inlier ratio
        inlier_layout = QHBoxLayout()
        inlier_layout.addWidget(QLabel("Inlier ratio:"))
        self.spin_inlier = QDoubleSpinBox()
        self.spin_inlier.setRange(0.01, 1.0)
        self.spin_inlier.setValue(0.1)
        self.spin_inlier.setSingleStep(0.05)
        self.spin_inlier.setToolTip("Minimum fraction of gradients supporting center")
        inlier_layout.addWidget(self.spin_inlier)
        ransac_layout.addLayout(inlier_layout)
        
        # Max error
        error_layout = QHBoxLayout()
        error_layout.addWidget(QLabel("Max error:"))
        self.spin_max_error = QDoubleSpinBox()
        self.spin_max_error.setRange(0.1, 5.0)
        self.spin_max_error.setValue(1.5)
        self.spin_max_error.setSingleStep(0.1)
        self.spin_max_error.setToolTip("Maximum allowed localization error")
        error_layout.addWidget(self.spin_max_error)
        ransac_layout.addLayout(error_layout)
        
        layout.addWidget(ransac_group)
        
        # Background subtraction
        bg_group = QGroupBox("Background")
        bg_layout = QVBoxLayout(bg_group)
        
        bg_combo_layout = QHBoxLayout()
        bg_combo_layout.addWidget(QLabel("Method:"))
        self.combo_background = QComboBox()
        self.combo_background.addItems([
            "None",
            "Mean",
            "Median",
            "RANSAC on Mean",
            "RANSAC on Median",
        ])
        self.combo_background.setToolTip("Background subtraction method")
        bg_combo_layout.addWidget(self.combo_background)
        bg_layout.addLayout(bg_combo_layout)
        
        layout.addWidget(bg_group)
        
        # Preview ROI section
        roi_group = QGroupBox("Preview ROI")
        roi_layout = QVBoxLayout(roi_group)
        
        self.btn_draw_roi = QPushButton("Draw ROI Rectangle")
        self.btn_draw_roi.setToolTip("Draw a rectangle to define preview region (10 Z-slices within XY bounds)")
        self.btn_draw_roi.clicked.connect(self._create_roi_layer)
        roi_layout.addWidget(self.btn_draw_roi)
        
        self.lbl_roi_status = QLabel("No ROI defined")
        self.lbl_roi_status.setStyleSheet("color: gray; font-size: 10px;")
        self.lbl_roi_status.setAlignment(Qt.AlignCenter)
        roi_layout.addWidget(self.lbl_roi_status)
        
        layout.addWidget(roi_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        self.btn_preview = QPushButton("Preview (ROI)")
        self.btn_preview.setToolTip("Run detection on ROI region (10 Z-slices centered on current position)")
        self.btn_preview.clicked.connect(self._run_preview)
        btn_layout.addWidget(self.btn_preview)
        
        self.btn_run = QPushButton("Run Full")
        self.btn_run.setToolTip("Run detection on entire 3D volume")
        self.btn_run.clicked.connect(self._run_full_detection)
        btn_layout.addWidget(self.btn_run)
        
        layout.addLayout(btn_layout)
        
        # Secondary buttons
        btn_layout2 = QHBoxLayout()
        
        self.btn_clear = QPushButton("Clear Results")
        self.btn_clear.setToolTip("Remove detected spots layers")
        self.btn_clear.clicked.connect(self._clear_results)
        btn_layout2.addWidget(self.btn_clear)
        
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setToolTip("Export detected spots to CSV file")
        self.btn_export.clicked.connect(self._export_results)
        btn_layout2.addWidget(self.btn_export)
        
        layout.addLayout(btn_layout2)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.lbl_status = QLabel("Select an image layer and adjust parameters")
        self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)
        
        # Log output (collapsible)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(100)
        self.txt_log.setVisible(False)
        layout.addWidget(self.txt_log)
        
        self.chk_show_log = QCheckBox("Show log")
        self.chk_show_log.stateChanged.connect(
            lambda s: self.txt_log.setVisible(s == Qt.Checked)
        )
        layout.addWidget(self.chk_show_log)
        
        layout.addStretch()
        
        # Disable controls if RS-FISH not found
        if not self.rsfish_path:
            self._set_controls_enabled(False)
    
    def _update_cli_status(self):
        """Update the CLI status indicator."""
        if self.rsfish_path:
            self.lbl_cli_status.setText(f"<small>CLI: {self.rsfish_path}</small>")
            self.lbl_cli_status.setStyleSheet("color: #4CAF50;")
        else:
            self.lbl_cli_status.setText("<small>RS-FISH CLI not found</small>")
            self.lbl_cli_status.setStyleSheet("color: red;")
    
    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable detection controls."""
        self.btn_preview.setEnabled(enabled)
        self.btn_run.setEnabled(enabled)
        self.btn_calc_anisotropy.setEnabled(enabled)
        self.btn_draw_roi.setEnabled(enabled)
    
    def _show_rsfish_not_found(self):
        """Show message about RS-FISH not being installed."""
        self.lbl_status.setText(
            "RS-FISH CLI not found. Please install it:\n"
            "git clone https://github.com/PreibischLab/RS-FISH.git\n"
            "cd RS-FISH && ./install $HOME/bin"
        )
        self.lbl_status.setStyleSheet("color: red;")
    
    def _connect_signals(self):
        """Connect to napari viewer signals."""
        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)
    
    def _on_layers_changed(self, event=None):
        """Handle layer list changes."""
        self._refresh_layer_list()
    
    def _refresh_layer_list(self):
        """Refresh the image layer combo box."""
        current = self.combo_image.currentText()
        
        self.combo_image.blockSignals(True)
        self.combo_image.clear()
        
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                # Skip our result layers
                if layer.name not in (self._points_layer_name, self._preview_layer_name):
                    self.combo_image.addItem(layer.name)
        
        # Restore selection if still valid
        idx = self.combo_image.findText(current)
        if idx >= 0:
            self.combo_image.setCurrentIndex(idx)
        
        self.combo_image.blockSignals(False)
    
    def _get_image_layer(self) -> Optional[Image]:
        """Get the currently selected image layer."""
        name = self.combo_image.currentText()
        if not name:
            return None
        
        for layer in self.viewer.layers:
            if layer.name == name and isinstance(layer, Image):
                return layer
        return None
    
    def _get_params(self) -> dict:
        """Get current parameter values as a dictionary."""
        return {
            'sigma': self.spin_sigma.value(),
            'threshold': self.spin_threshold.value(),
            'anisotropy': self.spin_anisotropy.value(),
            'support': self.spin_support.value(),
            'inlier_ratio': self.spin_inlier.value(),
            'max_error': self.spin_max_error.value(),
            'background': self.combo_background.currentIndex(),
            'auto_minmax': self.chk_auto_minmax.isChecked(),
        }
    
    def _get_temp_dir(self) -> str:
        """Get or create temporary directory."""
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="iviewer_rsfish_")
        return self._temp_dir.name
    
    def _create_roi_layer(self):
        """Create or activate the ROI shapes layer for drawing preview region."""
        # Check if ROI layer already exists
        roi_layer = None
        for layer in self.viewer.layers:
            if layer.name == self._roi_layer_name and isinstance(layer, Shapes):
                roi_layer = layer
                break
        
        if roi_layer is None:
            # Create new shapes layer
            roi_layer = self.viewer.add_shapes(
                name=self._roi_layer_name,
                edge_color='yellow',
                face_color='transparent',
                edge_width=2,
            )
        
        # Select the layer and set to rectangle drawing mode
        self.viewer.layers.selection.active = roi_layer
        roi_layer.mode = 'add_rectangle'
        
        self.lbl_status.setText("Draw a rectangle on the image to define preview ROI")
        self.lbl_status.setStyleSheet("color: #2196F3;")
        self.lbl_roi_status.setText("Drawing mode active...")
        self.lbl_roi_status.setStyleSheet("color: #2196F3; font-size: 10px;")
    
    def _get_roi_bounds(self):
        """Get the bounding box from the ROI shapes layer.
        
        Returns
        -------
        tuple or None
            (y_min, y_max, x_min, x_max) or None if no ROI defined
        """
        # Find ROI layer
        roi_layer = None
        for layer in self.viewer.layers:
            if layer.name == self._roi_layer_name and isinstance(layer, Shapes):
                roi_layer = layer
                break
        
        if roi_layer is None or len(roi_layer.data) == 0:
            return None
        
        # Get the last shape (most recently drawn)
        shape_data = roi_layer.data[-1]
        
        # shape_data is an array of vertices, get bounding box
        # For rectangle, vertices are the corners
        # Handle both 2D and 3D coordinates
        if shape_data.shape[1] == 2:
            # 2D: (y, x) coordinates
            y_coords = shape_data[:, 0]
            x_coords = shape_data[:, 1]
        else:
            # 3D: (z, y, x) coordinates - ignore z for XY ROI
            y_coords = shape_data[:, -2]
            x_coords = shape_data[:, -1]
        
        y_min, y_max = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))
        x_min, x_max = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
        
        return (y_min, y_max, x_min, x_max)
    
    def _run_preview(self):
        """Run RS-FISH on ROI region with 10 Z-slices centered on current position."""
        if not self.rsfish_path:
            self._show_rsfish_not_found()
            return
        
        image_layer = self._get_image_layer()
        if image_layer is None:
            self.lbl_status.setText("No image layer selected")
            self.lbl_status.setStyleSheet("color: orange;")
            return
        
        # Get ROI bounds
        roi_bounds = self._get_roi_bounds()
        if roi_bounds is None:
            self.lbl_status.setText("No ROI defined. Click 'Draw ROI Rectangle' first.")
            self.lbl_status.setStyleSheet("color: orange;")
            return
        
        y_min, y_max, x_min, x_max = roi_bounds
        data = image_layer.data
        
        # Validate bounds
        if data.ndim == 3:
            _, height, width = data.shape
        elif data.ndim == 2:
            height, width = data.shape
        else:
            self.lbl_status.setText(f"Unsupported image dimensions: {data.ndim}")
            self.lbl_status.setStyleSheet("color: red;")
            return
        
        # Clamp to image bounds
        y_min = max(0, y_min)
        y_max = min(height, y_max)
        x_min = max(0, x_min)
        x_max = min(width, x_max)
        
        if y_max <= y_min or x_max <= x_min:
            self.lbl_status.setText("Invalid ROI: zero size region")
            self.lbl_status.setStyleSheet("color: red;")
            return
        
        # Extract ROI with limited Z-slices for faster preview
        z_offset = 0
        if data.ndim == 3:
            n_slices = data.shape[0]
            preview_z_slices = 10  # Take 10 Z-slices for preview
            
            # Center around current Z position
            current_z = self.viewer.dims.current_step[0]
            half_slices = preview_z_slices // 2
            
            z_start = max(0, current_z - half_slices)
            z_end = min(n_slices, z_start + preview_z_slices)
            
            # Adjust start if we hit the end
            if z_end - z_start < preview_z_slices:
                z_start = max(0, z_end - preview_z_slices)
            
            preview_data = data[z_start:z_end, y_min:y_max, x_min:x_max]
            z_offset = z_start
            
            actual_z = z_end - z_start
            roi_size = f"{actual_z} x {y_max - y_min} x {x_max - x_min}"
        else:
            preview_data = data[y_min:y_max, x_min:x_max]
            roi_size = f"{y_max - y_min} x {x_max - x_min}"
        
        self.lbl_roi_status.setText(f"ROI: {roi_size} pixels")
        self.lbl_roi_status.setStyleSheet("color: #4CAF50; font-size: 10px;")
        
        # Run detection with offset information
        self._run_detection(
            preview_data, 
            is_preview=True, 
            z_offset=z_offset,
            y_offset=y_min,
            x_offset=x_min,
        )
    
    def _run_full_detection(self):
        """Run RS-FISH on entire volume."""
        if not self.rsfish_path:
            self._show_rsfish_not_found()
            return
        
        image_layer = self._get_image_layer()
        if image_layer is None:
            self.lbl_status.setText("No image layer selected")
            self.lbl_status.setStyleSheet("color: orange;")
            return
        
        self._run_detection(image_layer.data, is_preview=False)
    
    def _run_detection(
        self, 
        data: np.ndarray, 
        is_preview: bool = False, 
        z_offset: int = 0,
        y_offset: int = 0,
        x_offset: int = 0,
    ):
        """Run RS-FISH detection on the given data.
        
        Parameters
        ----------
        data : np.ndarray
            Image data to process (2D or 3D)
        is_preview : bool
            If True, this is a preview run
        z_offset : int
            Z-offset to add to results (for cropped regions)
        y_offset : int
            Y-offset to add to results (for ROI preview)
        x_offset : int
            X-offset to add to results (for ROI preview)
        """
        # Disable buttons during processing
        self.btn_preview.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        temp_dir = self._get_temp_dir()
        image_path = os.path.join(temp_dir, "input.tif")
        output_path = os.path.join(temp_dir, "output.csv")
        
        # Save image to temp file
        self.lbl_status.setText("Saving image to temporary file...")
        self.lbl_status.setStyleSheet("color: gray;")
        
        try:
            imwrite(image_path, data)
        except Exception as e:
            self.lbl_status.setText(f"Error saving image: {e}")
            self.lbl_status.setStyleSheet("color: red;")
            self._reset_ui()
            return
        
        # Get parameters
        params = self._get_params()
        
        # Create and start worker
        self._worker = RSFISHWorker(
            self.rsfish_path,
            image_path,
            output_path,
            params,
        )
        
        # Store metadata for result handling
        self._worker.is_preview = is_preview
        self._worker.z_offset = z_offset
        self._worker.y_offset = y_offset
        self._worker.x_offset = x_offset
        
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_detection_finished)
        self._worker.error.connect(self._on_detection_error)
        
        self.lbl_status.setText("Running RS-FISH detection...")
        self.lbl_status.setStyleSheet("color: gray;")
        
        self._worker.start()
    
    def _on_progress(self, message: str):
        """Handle progress messages from worker."""
        self.txt_log.append(message)
    
    def _on_detection_finished(self, df: pd.DataFrame):
        """Handle detection completion."""
        self._reset_ui()
        
        if df is None or len(df) == 0:
            self.lbl_status.setText("No spots detected")
            self.lbl_status.setStyleSheet("color: orange;")
            return
        
        is_preview = getattr(self._worker, 'is_preview', False)
        z_offset = getattr(self._worker, 'z_offset', 0)
        y_offset = getattr(self._worker, 'y_offset', 0)
        x_offset = getattr(self._worker, 'x_offset', 0)
        
        # Parse coordinates from DataFrame
        # RS-FISH output typically has columns: x, y, z (or similar)
        # Try common column name patterns
        coord_cols = None
        col_order = None  # Track if we need to reorder (z,y,x) vs (x,y,z)
        
        for pattern in [('z', 'y', 'x'), ('Z', 'Y', 'X'), ('axis-0', 'axis-1', 'axis-2')]:
            if all(c in df.columns for c in pattern):
                coord_cols = pattern
                col_order = 'zyx'
                break
        
        # Also check for x, y, z order (common in some RS-FISH outputs)
        if coord_cols is None:
            for pattern in [('x', 'y', 'z'), ('X', 'Y', 'Z')]:
                if all(c in df.columns for c in pattern):
                    coord_cols = pattern
                    col_order = 'xyz'
                    break
        
        if coord_cols is None:
            # Fall back to first 3 numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
            if len(numeric_cols) >= 3:
                coord_cols = tuple(numeric_cols)
                col_order = 'zyx'  # Assume zyx order
            else:
                self.lbl_status.setText(f"Could not parse coordinates from output. Columns: {list(df.columns)}")
                self.lbl_status.setStyleSheet("color: red;")
                return
        
        # Extract coordinates
        coords = df[list(coord_cols)].values.copy()
        
        # Convert to napari order (z, y, x) if needed
        if col_order == 'xyz':
            # Reorder from (x, y, z) to (z, y, x)
            coords = coords[:, [2, 1, 0]]
        
        # Apply offsets to map back to original image coordinates
        if z_offset > 0:
            coords[:, 0] += z_offset
        if y_offset > 0:
            coords[:, 1] += y_offset
        if x_offset > 0:
            coords[:, 2] += x_offset
        
        # Get intensity if available
        intensity_col = None
        for col in ['intensity', 'Intensity', 'int']:
            if col in df.columns:
                intensity_col = col
                break
        
        # Determine layer name
        layer_name = self._preview_layer_name if is_preview else self._points_layer_name
        
        # Remove existing layer if present
        for layer in list(self.viewer.layers):
            if layer.name == layer_name:
                self.viewer.layers.remove(layer)
                break
        
        # Add points layer
        face_color = 'cyan' if is_preview else 'red'
        size = 3 if is_preview else 4
        
        points_layer = self.viewer.add_points(
            coords,
            name=layer_name,
            face_color=face_color,
            size=size,
            blending='translucent',
        )
        
        # Store DataFrame as layer metadata
        points_layer.metadata['rsfish_df'] = df
        points_layer.metadata['params'] = self._get_params()
        
        n_spots = len(coords)
        mode_str = "preview" if is_preview else "full"
        self.lbl_status.setText(f"Detected {n_spots} spots ({mode_str})")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
    
    def _on_detection_error(self, error_msg: str):
        """Handle detection error."""
        self._reset_ui()
        self.lbl_status.setText(f"Error: {error_msg}")
        self.lbl_status.setStyleSheet("color: red;")
        self.txt_log.append(f"ERROR: {error_msg}")
    
    def _reset_ui(self):
        """Reset UI after detection completes."""
        self.btn_preview.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._worker = None
    
    def _clear_results(self):
        """Remove detected spots layers and ROI."""
        for layer_name in (self._points_layer_name, self._preview_layer_name, self._roi_layer_name):
            for layer in list(self.viewer.layers):
                if layer.name == layer_name:
                    self.viewer.layers.remove(layer)
        
        self.lbl_status.setText("Results cleared")
        self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
        self.lbl_roi_status.setText("No ROI defined")
        self.lbl_roi_status.setStyleSheet("color: gray; font-size: 10px;")
    
    def _export_results(self):
        """Export detected spots to CSV file."""
        # Find the full detection layer
        points_layer = None
        for layer in self.viewer.layers:
            if layer.name == self._points_layer_name and isinstance(layer, Points):
                points_layer = layer
                break
        
        if points_layer is None:
            self.lbl_status.setText("No detection results to export")
            self.lbl_status.setStyleSheet("color: orange;")
            return
        
        # Get DataFrame from metadata or create from coordinates
        if 'rsfish_df' in points_layer.metadata:
            df = points_layer.metadata['rsfish_df']
        else:
            coords = points_layer.data
            df = pd.DataFrame(coords, columns=['z', 'y', 'x'])
        
        # Ask for save location
        from qtpy.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Detections",
            "rsfish_detections.csv",
            "CSV files (*.csv)",
        )
        
        if filepath:
            df.to_csv(filepath, index=False)
            self.lbl_status.setText(f"Exported {len(df)} spots to {Path(filepath).name}")
            self.lbl_status.setStyleSheet("color: #4CAF50;")
    
    def _calculate_anisotropy(self):
        """Calculate anisotropy coefficient using rs-fish-anisotropy."""
        # Check for rs-fish-anisotropy
        aniso_path = shutil.which("rs-fish-anisotropy")
        if not aniso_path:
            # Try common paths
            for path in [
                Path.home() / "bin" / "rs-fish-anisotropy",
                Path.home() / ".local" / "bin" / "rs-fish-anisotropy",
            ]:
                if path.exists():
                    aniso_path = str(path)
                    break
        
        if not aniso_path:
            QMessageBox.warning(
                self,
                "Tool Not Found",
                "rs-fish-anisotropy not found.\n\n"
                "Please install RS-FISH CLI tools:\n"
                "git clone https://github.com/PreibischLab/RS-FISH.git\n"
                "cd RS-FISH && ./install $HOME/bin"
            )
            return
        
        image_layer = self._get_image_layer()
        if image_layer is None:
            QMessageBox.warning(self, "No Image", "Please select an image layer first.")
            return
        
        # For now, show a message that this feature requires manual calculation
        QMessageBox.information(
            self,
            "Calculate Anisotropy",
            "Anisotropy calculation requires interactive selection of spots.\n\n"
            "For now, please calculate it manually:\n"
            "1. Open your image in Fiji\n"
            "2. Run Plugins > RS-FISH > Tools > Calculate Anisotropy Coefficient\n"
            "3. Enter the resulting value here\n\n"
            "Typical values are 1.0 (isotropic) to 3.0 (highly anisotropic)."
        )
    
    def closeEvent(self, event):
        """Clean up when widget is closed."""
        # Clean up temp directory
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
        
        # Stop worker if running
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        
        super().closeEvent(event)
