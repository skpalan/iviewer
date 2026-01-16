"""
Mask Highlight Widget for napari

Provides functionality to highlight layer content based on a selected mask:
- Choose a mask layer to use for highlighting
- Adjust transparency of content outside the mask
- Adjust color/tint of content outside the mask
"""

from typing import Dict, Optional, Tuple

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSlider,
    QPushButton,
    QFrame,
    QColorDialog,
    QCheckBox,
    QSpinBox,
    QGroupBox,
)
from qtpy.QtGui import QColor

import napari
from napari.layers import Image, Labels, Points


class MaskHighlightWidget(QWidget):
    """Widget for highlighting layer content based on mask layers.
    
    Features:
    - Select a target layer to apply masking effect
    - Choose a mask layer (Labels layer) for highlighting
    - Adjust transparency of content outside the mask
    - Set a color tint for content outside the mask
    - Option to highlight specific label IDs within the mask
    
    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer instance.
    """
    
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # Store original layer data for restoration
        self._original_data: Dict[str, np.ndarray] = {}
        
        # Store current highlight settings per layer
        self._highlight_settings: Dict[str, dict] = {}
        
        # Track overlay layers we create
        self._overlay_layers: Dict[str, str] = {}  # target_layer -> overlay_layer_name
        
        self._setup_ui()
        self._connect_signals()
        self._refresh_layer_lists()
    
    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header
        header = QLabel("<b>Mask Highlight</b>")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Target layer selection
        target_group = QGroupBox("Target Layer")
        target_layout = QVBoxLayout(target_group)
        
        self.combo_target = QComboBox()
        self.combo_target.setToolTip("Select the layer to apply mask highlighting")
        self.combo_target.currentTextChanged.connect(self._on_target_changed)
        target_layout.addWidget(self.combo_target)
        
        layout.addWidget(target_group)
        
        # Mask layer selection
        mask_group = QGroupBox("Mask Layer")
        mask_layout = QVBoxLayout(mask_group)
        
        self.combo_mask = QComboBox()
        self.combo_mask.setToolTip("Select a Labels layer to use as mask")
        self.combo_mask.addItem("(None)")
        mask_layout.addWidget(self.combo_mask)
        
        # Label ID selection
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label ID:"))
        self.spin_label_id = QSpinBox()
        self.spin_label_id.setMinimum(0)
        self.spin_label_id.setMaximum(99999)
        self.spin_label_id.setValue(0)
        self.spin_label_id.setToolTip("0 = all labels, or specify a label ID to highlight")
        self.spin_label_id.setSpecialValueText("All")
        label_layout.addWidget(self.spin_label_id)
        mask_layout.addLayout(label_layout)
        
        # Invert mask option
        self.chk_invert = QCheckBox("Invert mask (highlight outside)")
        self.chk_invert.setToolTip("If checked, highlight areas outside the mask instead")
        mask_layout.addWidget(self.chk_invert)
        
        layout.addWidget(mask_group)
        
        # Outside mask appearance
        appear_group = QGroupBox("Outside Mask Appearance")
        appear_layout = QVBoxLayout(appear_group)
        
        # Opacity slider
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setMinimum(0)
        self.slider_opacity.setMaximum(100)
        self.slider_opacity.setValue(30)
        self.slider_opacity.setToolTip("Opacity of content outside the mask (0=invisible, 100=fully visible)")
        opacity_layout.addWidget(self.slider_opacity)
        self.lbl_opacity = QLabel("30%")
        self.lbl_opacity.setMinimumWidth(40)
        opacity_layout.addWidget(self.lbl_opacity)
        appear_layout.addLayout(opacity_layout)
        
        self.slider_opacity.valueChanged.connect(
            lambda v: self.lbl_opacity.setText(f"{v}%")
        )
        
        # Color tint
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Tint color:"))
        self.btn_color = QPushButton()
        self.btn_color.setFixedSize(60, 25)
        self._outside_color = QColor(0, 0, 0)  # Default black
        self._update_color_button()
        self.btn_color.clicked.connect(self._pick_color)
        self.btn_color.setToolTip("Color to tint content outside the mask")
        color_layout.addWidget(self.btn_color)
        color_layout.addStretch()
        
        self.chk_use_tint = QCheckBox("Apply tint")
        self.chk_use_tint.setChecked(False)
        self.chk_use_tint.setToolTip("Apply color tint to outside areas (otherwise just dims)")
        color_layout.addWidget(self.chk_use_tint)
        appear_layout.addLayout(color_layout)
        
        # Points-specific option: hide edge/stroke for outside points
        self.chk_hide_edge = QCheckBox("Hide edge/stroke for outside points")
        self.chk_hide_edge.setChecked(True)
        self.chk_hide_edge.setToolTip("Remove the outline/stroke from points outside the mask (Points layers only)")
        appear_layout.addWidget(self.chk_hide_edge)
        
        layout.addWidget(appear_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setToolTip("Apply mask highlighting to target layer")
        self.btn_apply.clicked.connect(self._apply_highlight)
        btn_layout.addWidget(self.btn_apply)
        
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Remove highlighting and restore original layer")
        self.btn_reset.clicked.connect(self._reset_highlight)
        btn_layout.addWidget(self.btn_reset)
        
        layout.addLayout(btn_layout)
        
        # Live preview option
        self.chk_live = QCheckBox("Live preview")
        self.chk_live.setChecked(False)
        self.chk_live.setToolTip("Automatically update when settings change")
        self.chk_live.stateChanged.connect(self._on_live_changed)
        layout.addWidget(self.chk_live)
        
        # Status label
        self.lbl_status = QLabel("Select a target layer and mask")
        self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)
        
        # Instructions
        instructions = QLabel(
            "<small>• Select target layer to highlight<br>"
            "• Choose a mask (Labels) layer<br>"
            "• Adjust outside appearance<br>"
            "• Click Apply to see effect</small>"
        )
        instructions.setStyleSheet("color: gray;")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
        
        layout.addStretch()
    
    def _connect_signals(self):
        """Connect to napari viewer signals."""
        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)
        self.viewer.layers.events.reordered.connect(self._on_layers_changed)
        
        # Connect settings changes for live preview
        self.combo_mask.currentTextChanged.connect(self._on_settings_changed)
        self.spin_label_id.valueChanged.connect(self._on_settings_changed)
        self.chk_invert.stateChanged.connect(self._on_settings_changed)
        self.slider_opacity.valueChanged.connect(self._on_settings_changed)
        self.chk_use_tint.stateChanged.connect(self._on_settings_changed)
        self.chk_hide_edge.stateChanged.connect(self._on_settings_changed)
    
    def _on_layers_changed(self, event=None):
        """Handle layer list changes."""
        self._refresh_layer_lists()
    
    def _refresh_layer_lists(self):
        """Refresh the layer combo boxes."""
        current_target = self.combo_target.currentText()
        current_mask = self.combo_mask.currentText()
        
        self.combo_target.blockSignals(True)
        self.combo_mask.blockSignals(True)
        
        self.combo_target.clear()
        self.combo_mask.clear()
        self.combo_mask.addItem("(None)")
        
        for layer in self.viewer.layers:
            # Skip our overlay layers
            if layer.name.endswith(" [highlighted]"):
                continue
            
            # Target can be Image or Points layers
            if isinstance(layer, (Image, Points)):
                self.combo_target.addItem(layer.name)
            
            # Mask must be Labels layer
            if isinstance(layer, Labels):
                self.combo_mask.addItem(layer.name)
        
        # Restore previous selections if still valid
        idx = self.combo_target.findText(current_target)
        if idx >= 0:
            self.combo_target.setCurrentIndex(idx)
        
        idx = self.combo_mask.findText(current_mask)
        if idx >= 0:
            self.combo_mask.setCurrentIndex(idx)
        
        self.combo_target.blockSignals(False)
        self.combo_mask.blockSignals(False)
    
    def _on_target_changed(self, text: str):
        """Handle target layer selection change."""
        if not text:
            return
        
        # Check if this layer has stored settings
        if text in self._highlight_settings:
            settings = self._highlight_settings[text]
            self.combo_mask.setCurrentText(settings.get('mask', '(None)'))
            self.spin_label_id.setValue(settings.get('label_id', 0))
            self.chk_invert.setChecked(settings.get('invert', False))
            self.slider_opacity.setValue(settings.get('opacity', 30))
            self.chk_use_tint.setChecked(settings.get('use_tint', False))
    
    def _on_settings_changed(self):
        """Handle settings change - apply if live preview is on."""
        if self.chk_live.isChecked():
            self._apply_highlight()
    
    def _on_live_changed(self, state: int):
        """Handle live preview checkbox change."""
        if state == Qt.Checked:
            self._apply_highlight()
    
    def _update_color_button(self):
        """Update the color button appearance."""
        self.btn_color.setStyleSheet(
            f"background-color: {self._outside_color.name()}; border: 1px solid gray;"
        )
    
    def _pick_color(self):
        """Open color picker dialog."""
        color = QColorDialog.getColor(self._outside_color, self, "Select Tint Color")
        if color.isValid():
            self._outside_color = color
            self._update_color_button()
            self._on_settings_changed()
    
    def _get_target_layer(self):
        """Get the currently selected target layer (Image or Points)."""
        name = self.combo_target.currentText()
        if not name:
            return None
        
        for layer in self.viewer.layers:
            if layer.name == name and isinstance(layer, (Image, Points)):
                return layer
        return None
    
    def _get_mask_layer(self) -> Optional[Labels]:
        """Get the currently selected mask layer."""
        name = self.combo_mask.currentText()
        if not name or name == "(None)":
            return None
        
        for layer in self.viewer.layers:
            if layer.name == name and isinstance(layer, Labels):
                return layer
        return None
    
    def _apply_highlight(self):
        """Apply mask highlighting to the target layer."""
        target_layer = self._get_target_layer()
        if target_layer is None:
            self.lbl_status.setText("No target layer selected")
            self.lbl_status.setStyleSheet("color: orange;")
            return
        
        mask_layer = self._get_mask_layer()
        if mask_layer is None:
            self.lbl_status.setText("No mask layer selected")
            self.lbl_status.setStyleSheet("color: orange;")
            return
        
        target_name = target_layer.name
        
        # Get settings
        label_id = self.spin_label_id.value()
        invert = self.chk_invert.isChecked()
        opacity = self.slider_opacity.value() / 100.0
        use_tint = self.chk_use_tint.isChecked()
        tint_color = (
            self._outside_color.red(),
            self._outside_color.green(),
            self._outside_color.blue(),
        )
        
        # Store settings
        self._highlight_settings[target_name] = {
            'mask': self.combo_mask.currentText(),
            'label_id': label_id,
            'invert': invert,
            'opacity': int(opacity * 100),
            'use_tint': use_tint,
            'tint_color': tint_color,
        }
        
        try:
            mask_data = mask_layer.data
            
            # Create the mask (binary)
            if label_id == 0:
                # All non-zero labels
                binary_mask = mask_data > 0
            else:
                # Specific label
                binary_mask = mask_data == label_id
            
            # Invert if requested
            if invert:
                binary_mask = ~binary_mask
            
            # Handle differently based on layer type
            if isinstance(target_layer, Points):
                hide_edge = self.chk_hide_edge.isChecked()
                self._apply_highlight_to_points(
                    target_layer, target_name, binary_mask, 
                    opacity, use_tint, tint_color, hide_edge
                )
            else:
                # Image layer
                self._apply_highlight_to_image(
                    target_layer, target_name, binary_mask,
                    opacity, use_tint, tint_color
                )
            
            self.lbl_status.setText(f"Highlighting applied to '{target_name}'")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
        except Exception as e:
            self.lbl_status.setText(f"Error: {str(e)}")
            self.lbl_status.setStyleSheet("color: red;")
            print(f"Mask highlight error: {e}")
            import traceback
            traceback.print_exc()
    
    def _apply_highlight_to_image(
        self, 
        target_layer: Image, 
        target_name: str,
        binary_mask: np.ndarray,
        opacity: float,
        use_tint: bool,
        tint_color: Tuple[int, int, int],
    ):
        """Apply mask highlight effect to an Image layer."""
        # Store original data if not already stored
        if target_name not in self._original_data:
            self._original_data[target_name] = target_layer.data.copy()
        
        original = self._original_data[target_name]
        
        # Handle shape mismatch - resize mask if needed
        if binary_mask.shape != original.shape:
            binary_mask = self._resize_mask(binary_mask, original.shape)
        
        # Apply the highlight effect
        highlighted = self._apply_mask_effect(
            original, binary_mask, opacity, use_tint, tint_color
        )
        
        # Update the layer data
        target_layer.data = highlighted
    
    def _apply_highlight_to_points(
        self,
        target_layer: Points,
        target_name: str,
        binary_mask: np.ndarray,
        opacity: float,
        use_tint: bool,
        tint_color: Tuple[int, int, int],
        hide_edge: bool,
    ):
        """Apply mask highlight effect to a Points layer.
        
        Points outside the mask will have their color changed based on settings.
        """
        # Store original face_color and border_color if not already stored
        if target_name not in self._original_data:
            # For points, store both face_color and border_color arrays
            self._original_data[target_name] = {
                'face_color': target_layer.face_color.copy(),
                'border_color': target_layer.border_color.copy(),
            }
        
        original_face = self._original_data[target_name]['face_color']
        original_border = self._original_data[target_name]['border_color']
        coords = target_layer.data  # (N, ndim) array of coordinates
        
        # Determine which points are inside/outside the mask
        n_points = len(coords)
        inside_mask = np.zeros(n_points, dtype=bool)
        
        mask_shape = np.array(binary_mask.shape)
        
        for i, coord in enumerate(coords):
            # Convert coordinates to integer indices
            # Coordinates are in (z, y, x) or (y, x) order
            idx = tuple(int(round(c)) for c in coord)
            
            # Check bounds
            in_bounds = all(0 <= idx[j] < mask_shape[j] for j in range(len(idx)))
            
            if in_bounds:
                # Handle dimension mismatch (2D mask for 3D points)
                if len(idx) > len(mask_shape):
                    # Use only the last dimensions that match the mask
                    idx = idx[-len(mask_shape):]
                elif len(idx) < len(mask_shape):
                    # Use only the first dimensions of the mask
                    idx = idx + (0,) * (len(mask_shape) - len(idx))
                
                try:
                    inside_mask[i] = binary_mask[idx]
                except IndexError:
                    inside_mask[i] = False
            else:
                inside_mask[i] = False
        
        # Create new colors arrays
        new_face_colors = original_face.copy()
        new_border_colors = original_border.copy()
        
        # Modify colors for points outside the mask
        outside_indices = ~inside_mask
        
        # Always apply opacity to alpha channel for outside points
        new_face_colors[outside_indices, 3] = original_face[outside_indices, 3] * opacity
        
        if use_tint:
            # Also blend RGB channels with tint color for outside points
            tint_rgb = np.array([
                tint_color[0] / 255.0,
                tint_color[1] / 255.0,
                tint_color[2] / 255.0,
            ])
            # Blend original RGB with tint (50/50 blend for visible tinting)
            new_face_colors[outside_indices, :3] = (
                original_face[outside_indices, :3] * 0.5 +
                tint_rgb * 0.5
            )
        
        # Handle border/stroke visibility for outside points
        if hide_edge:
            # Make border transparent for outside points
            new_border_colors[outside_indices, 3] = 0.0
        
        # Update the layer colors
        target_layer.face_color = new_face_colors
        target_layer.border_color = new_border_colors
        
        # Update status with point counts
        n_inside = np.sum(inside_mask)
        n_outside = np.sum(outside_indices)
        print(f"Points: {n_inside} inside mask, {n_outside} outside mask")
    
    def _resize_mask(self, mask: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize mask to match target shape using nearest neighbor.
        
        Handles common cases:
        - 2D mask for 3D data (broadcast along Z)
        - Different XY dimensions (resize)
        """
        from scipy import ndimage
        
        # If mask is 2D and target is 3D, broadcast
        if mask.ndim == 2 and len(target_shape) == 3:
            # Resize 2D mask to match XY dimensions
            zoom_factors = (
                target_shape[1] / mask.shape[0],
                target_shape[2] / mask.shape[1],
            )
            resized_2d = ndimage.zoom(mask.astype(float), zoom_factors, order=0) > 0.5
            # Broadcast to 3D
            return np.broadcast_to(resized_2d, target_shape).copy()
        
        # If same number of dimensions, resize
        if mask.ndim == len(target_shape):
            zoom_factors = tuple(t / m for t, m in zip(target_shape, mask.shape))
            return ndimage.zoom(mask.astype(float), zoom_factors, order=0) > 0.5
        
        # If mask is 3D and target is 3D but different sizes
        if mask.ndim == 3 and len(target_shape) == 3:
            zoom_factors = tuple(t / m for t, m in zip(target_shape, mask.shape))
            return ndimage.zoom(mask.astype(float), zoom_factors, order=0) > 0.5
        
        raise ValueError(
            f"Cannot resize mask of shape {mask.shape} to match {target_shape}"
        )
    
    def _apply_mask_effect(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        opacity: float,
        use_tint: bool,
        tint_color: Tuple[int, int, int],
    ) -> np.ndarray:
        """Apply the mask effect to image data.
        
        Parameters
        ----------
        data : np.ndarray
            Original image data.
        mask : np.ndarray
            Binary mask (True = inside mask/highlighted area).
        opacity : float
            Opacity for outside mask (0-1).
        use_tint : bool
            Whether to apply color tint to outside areas.
        tint_color : tuple
            RGB color for tinting (0-255 each).
            
        Returns
        -------
        np.ndarray
            Modified image data with mask effect applied.
        """
        result = data.copy().astype(np.float32)
        
        # Areas inside mask stay as-is, areas outside get modified
        outside_mask = ~mask
        
        if use_tint and data.ndim >= 3 and data.shape[-1] in (3, 4):
            # RGB/RGBA image - apply color tint
            for i, c in enumerate(tint_color):
                channel = result[..., i]
                channel[outside_mask] = (
                    channel[outside_mask] * opacity + 
                    c * (1 - opacity)
                )
        else:
            # Grayscale or non-RGB - just apply opacity (darken)
            result[outside_mask] = result[outside_mask] * opacity
        
        # Convert back to original dtype
        if np.issubdtype(data.dtype, np.integer):
            result = np.clip(result, 0, np.iinfo(data.dtype).max)
            result = result.astype(data.dtype)
        else:
            result = result.astype(data.dtype)
        
        return result
    
    def _reset_highlight(self):
        """Reset the target layer to original data/colors."""
        target_layer = self._get_target_layer()
        if target_layer is None:
            self.lbl_status.setText("No target layer selected")
            self.lbl_status.setStyleSheet("color: orange;")
            return
        
        target_name = target_layer.name
        
        if target_name in self._original_data:
            # Restore based on layer type
            if isinstance(target_layer, Points):
                # For Points, restore face_color and border_color
                original = self._original_data[target_name]
                target_layer.face_color = original['face_color']
                target_layer.border_color = original['border_color']
            else:
                # For Image, restore data
                target_layer.data = self._original_data[target_name]
            
            del self._original_data[target_name]
            
            if target_name in self._highlight_settings:
                del self._highlight_settings[target_name]
            
            self.lbl_status.setText(f"Reset '{target_name}' to original")
            self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
        else:
            self.lbl_status.setText("No changes to reset")
            self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
    
    def reset_all(self):
        """Reset all layers to their original data/colors."""
        for layer_name, original_data in list(self._original_data.items()):
            for layer in self.viewer.layers:
                if layer.name == layer_name:
                    if isinstance(layer, Points):
                        # For Points, restore face_color and border_color
                        layer.face_color = original_data['face_color']
                        layer.border_color = original_data['border_color']
                    else:
                        # For Image, restore data
                        layer.data = original_data
                    break
        
        self._original_data.clear()
        self._highlight_settings.clear()
        self.lbl_status.setText("All layers reset")
        self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
