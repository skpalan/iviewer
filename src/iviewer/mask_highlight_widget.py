"""
Mask Highlight Widget for napari

Provides functionality to highlight layer content based on a selected mask:
- Choose a mask layer to use for highlighting
- Adjust transparency of content outside the mask
- Adjust color/tint of content outside the mask
- List all cells in mask with statistics (voxel count, centroid)
- Select and highlight specific cells in 2D/3D view
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import ndimage
from qtpy.QtCore import Qt, QTimer
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
    QApplication,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
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
    - List all cells in mask with statistics (ID, voxel count, centroid)
    - Select and highlight specific cells in both 2D and 3D view modes
    
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
        
        # Timer for debouncing grid refresh (prevents race conditions)
        self._grid_refresh_timer: Optional['QTimer'] = None
        
        # Track layer names by layer ID for detecting renames
        self._layer_names: Dict[int, str] = {}
        
        # Store callbacks for layer name change events
        self._layer_name_callbacks: Dict[int, callable] = {}
        
        # Cell list state
        self._cell_ids: List[int] = []  # All cell IDs in current mask
        self._cell_stats: Dict[int, dict] = {}  # {cell_id: {'voxels': int, 'centroid': tuple}}
        self._selected_cell_ids: Set[int] = set()  # Currently selected cells
        self._cell_selection_overlay: Optional[str] = None  # Name of overlay layer
        self._original_mask_opacity: float = 0.5  # For restoration
        
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
        
        # Cell List section
        cell_list_group = QGroupBox("Cell List")
        cell_list_layout = QVBoxLayout(cell_list_group)
        
        # Action buttons for cell list
        cell_btn_layout = QHBoxLayout()
        
        self.btn_refresh_cells = QPushButton("Refresh")
        self.btn_refresh_cells.setToolTip("Refresh cell list from mask layer")
        self.btn_refresh_cells.clicked.connect(self._refresh_cell_list)
        cell_btn_layout.addWidget(self.btn_refresh_cells)
        
        self.btn_select_all_cells = QPushButton("Select All")
        self.btn_select_all_cells.setToolTip("Select all cells")
        self.btn_select_all_cells.clicked.connect(self._select_all_cells)
        cell_btn_layout.addWidget(self.btn_select_all_cells)
        
        self.btn_clear_cell_selection = QPushButton("Clear")
        self.btn_clear_cell_selection.setToolTip("Clear cell selection")
        self.btn_clear_cell_selection.clicked.connect(self._clear_cell_selection)
        cell_btn_layout.addWidget(self.btn_clear_cell_selection)
        
        cell_list_layout.addLayout(cell_btn_layout)
        
        # Table widget for cell list
        self.table_cells = QTableWidget()
        self.table_cells.setColumnCount(3)
        self.table_cells.setHorizontalHeaderLabels(["ID", "Voxels", "Centroid (Z, Y, X)"])
        self.table_cells.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_cells.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table_cells.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_cells.horizontalHeader().setStretchLastSection(True)
        self.table_cells.setColumnWidth(0, 60)
        self.table_cells.setColumnWidth(1, 80)
        self.table_cells.setFixedHeight(200)
        self.table_cells.verticalScrollBar().valueChanged.connect(self._on_table_scroll)
        self.table_cells.itemSelectionChanged.connect(self._on_cell_selection_changed)
        cell_list_layout.addWidget(self.table_cells)
        
        # Cell count label
        self.lbl_cell_count = QLabel("Selected: 0 | Total: 0 cells")
        self.lbl_cell_count.setStyleSheet("color: gray; font-style: italic;")
        cell_list_layout.addWidget(self.lbl_cell_count)
        
        # Highlight buttons
        highlight_btn_layout = QHBoxLayout()
        
        self.btn_highlight_cells = QPushButton("Highlight Selected")
        self.btn_highlight_cells.setToolTip("Highlight selected cells in 2D/3D view")
        self.btn_highlight_cells.clicked.connect(self._highlight_selected_cells)
        highlight_btn_layout.addWidget(self.btn_highlight_cells)
        
        self.btn_clear_highlight = QPushButton("Clear Highlight")
        self.btn_clear_highlight.setToolTip("Remove cell highlighting")
        self.btn_clear_highlight.clicked.connect(self._clear_cell_highlight)
        highlight_btn_layout.addWidget(self.btn_clear_highlight)
        
        cell_list_layout.addLayout(highlight_btn_layout)
        
        layout.addWidget(cell_list_group)
        
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
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.layers.events.reordered.connect(self._on_layers_changed)
        
        # Connect to existing layers' name change events
        for layer in self.viewer.layers:
            self._connect_layer_name_signal(layer)
        
        # Connect settings changes for live preview
        self.combo_mask.currentTextChanged.connect(self._on_settings_changed)
        self.spin_label_id.valueChanged.connect(self._on_settings_changed)
        self.chk_invert.stateChanged.connect(self._on_settings_changed)
        self.slider_opacity.valueChanged.connect(self._on_settings_changed)
        self.chk_use_tint.stateChanged.connect(self._on_settings_changed)
        self.chk_hide_edge.stateChanged.connect(self._on_settings_changed)
        
        # Connect mask changes to refresh cell list
        self.combo_mask.currentTextChanged.connect(self._refresh_cell_list)
    
    def _connect_layer_name_signal(self, layer):
        """Connect to a layer's name change event and track its name."""
        layer_id = id(layer)
        
        # Store the current name for tracking renames
        self._layer_names[layer_id] = layer.name
        
        # Create a closure to capture the layer reference
        def on_name_change(event):
            self._on_layer_name_changed(layer)
        
        # Store the callback reference so we can disconnect later if needed
        self._layer_name_callbacks[layer_id] = on_name_change
        
        # Connect to the name change event
        layer.events.name.connect(on_name_change)
    
    def _on_layer_inserted(self, event):
        """Handle layer added to viewer."""
        layer = event.value
        self._connect_layer_name_signal(layer)
        self._refresh_layer_lists()
    
    def _on_layer_removed(self, event):
        """Handle layer removed from viewer."""
        layer = event.value
        layer_id = id(layer)
        
        # Clean up tracking data
        if layer_id in self._layer_name_callbacks:
            del self._layer_name_callbacks[layer_id]
        if layer_id in self._layer_names:
            del self._layer_names[layer_id]
        
        self._refresh_layer_lists()
    
    def _on_layer_name_changed(self, layer):
        """Handle layer name change - update internal dictionaries.
        
        This ensures that stored original data and settings are accessible
        after a layer is renamed (e.g., by Tidy Layers or manual rename).
        """
        layer_id = id(layer)
        new_name = layer.name
        
        # Get the old name from our tracking dict
        old_name = self._layer_names.get(layer_id)
        
        if old_name is None or old_name == new_name:
            # No tracked name or name hasn't actually changed
            return
        
        # Update our tracking to the new name
        self._layer_names[layer_id] = new_name
        
        # Update _original_data if the old name was stored
        if old_name in self._original_data:
            self._original_data[new_name] = self._original_data.pop(old_name)
        
        # Update _highlight_settings if the old name was stored
        if old_name in self._highlight_settings:
            self._highlight_settings[new_name] = self._highlight_settings.pop(old_name)
        
        # Update _overlay_layers if the old name was stored
        if old_name in self._overlay_layers:
            self._overlay_layers[new_name] = self._overlay_layers.pop(old_name)
        
        # Refresh the combo boxes to show the new name
        self._refresh_layer_lists()
    
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
            
            # Refresh grid mode if active to fix potential display issues
            self._refresh_grid_mode()
            
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
            
            # Refresh grid mode if active
            self._refresh_grid_mode()
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
        
        # Refresh grid mode if active
        self._refresh_grid_mode()
    
    def _refresh_cell_list(self):
        """Populate cell list table from selected mask layer."""
        mask_layer = self._get_mask_layer()
        
        # Clear table and state
        self.table_cells.setRowCount(0)
        self._cell_ids = []
        self._cell_stats = {}
        self._selected_cell_ids.clear()
        self._update_cell_count_label()
        
        if mask_layer is None:
            return
        
        # Get unique cell IDs (excluding background 0)
        unique_ids = np.unique(mask_layer.data)
        self._cell_ids = [int(cid) for cid in unique_ids if cid > 0]
        
        # For small datasets (< 500 cells), compute all stats upfront
        # For larger datasets, use lazy loading
        if len(self._cell_ids) < 500:
            self._compute_all_cell_stats(mask_layer.data)
        
        # Populate table rows
        self.table_cells.setRowCount(len(self._cell_ids))
        for row, cell_id in enumerate(self._cell_ids):
            # ID column
            id_item = QTableWidgetItem(str(cell_id))
            id_item.setData(Qt.UserRole, cell_id)
            self.table_cells.setItem(row, 0, id_item)
            
            # Check if stats already computed
            if cell_id in self._cell_stats:
                stats = self._cell_stats[cell_id]
                self.table_cells.setItem(row, 1, QTableWidgetItem(str(stats['voxels'])))
                centroid = stats['centroid']
                if len(centroid) == 3:
                    centroid_str = f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"
                elif len(centroid) == 2:
                    centroid_str = f"({centroid[0]:.1f}, {centroid[1]:.1f})"
                else:
                    centroid_str = str(centroid)
                self.table_cells.setItem(row, 2, QTableWidgetItem(centroid_str))
            else:
                # Placeholder for lazy loading
                self.table_cells.setItem(row, 1, QTableWidgetItem("..."))
                self.table_cells.setItem(row, 2, QTableWidgetItem("..."))
        
        self._update_cell_count_label()
        
        # For large datasets, compute stats for visible rows
        if len(self._cell_ids) >= 500:
            self._compute_visible_cell_stats()
    
    def _compute_all_cell_stats(self, mask_data):
        """Compute stats for all cells at once using vectorized operations.
        
        This method uses scipy.ndimage.center_of_mass and numpy.unique to
        compute all cell statistics in O(M) time where M is the number of
        voxels, rather than O(N × M) where N is the number of cells.
        
        Parameters
        ----------
        mask_data : np.ndarray
            The mask layer data array
        """
        # Compute voxel counts for all labels in one pass using unique
        unique_labels, counts = np.unique(mask_data, return_counts=True)
        label_to_count = dict(zip(unique_labels, counts))
        
        # Compute centroids for all cells in one pass using scipy
        # center_of_mass returns centroids for all specified labels at once
        centroids = ndimage.center_of_mass(
            np.ones_like(mask_data),  # Input (uniform mass)
            labels=mask_data,          # Label array
            index=self._cell_ids       # Compute for these specific labels
        )
        
        # Handle case where only one cell is present (returns single tuple instead of list)
        if len(self._cell_ids) == 1:
            centroids = [centroids]
        
        # Store results for all cells
        for i, cell_id in enumerate(self._cell_ids):
            voxel_count = label_to_count.get(cell_id, 0)
            
            # Convert centroid to tuple if it's an array/list
            centroid = centroids[i]
            if not isinstance(centroid, tuple):
                centroid = tuple(float(c) for c in centroid)
            
            self._cell_stats[cell_id] = {
                'voxels': voxel_count,
                'centroid': centroid,
            }
    
    def _compute_visible_cell_stats(self):
        """Compute stats for currently visible table rows (lazy loading for large datasets)."""
        mask_layer = self._get_mask_layer()
        if mask_layer is None or len(self._cell_ids) == 0:
            return
        
        # Get visible row range
        first_visible = self.table_cells.rowAt(0)
        last_visible = self.table_cells.rowAt(self.table_cells.viewport().height())
        
        if first_visible < 0:
            first_visible = 0
        if last_visible < 0:
            last_visible = len(self._cell_ids) - 1
        
        # Compute stats for visible cells that haven't been computed
        mask_data = mask_layer.data
        for row in range(first_visible, min(last_visible + 1, len(self._cell_ids))):
            cell_id = self._cell_ids[row]
            if cell_id in self._cell_stats:
                continue  # Already computed
            
            # Calculate stats
            mask = mask_data == cell_id
            voxel_count = int(np.sum(mask))
            
            # Centroid via np.where and mean
            coords = np.where(mask)
            centroid = tuple(float(np.mean(c)) for c in coords)
            
            self._cell_stats[cell_id] = {
                'voxels': voxel_count,
                'centroid': centroid,
            }
            
            # Update table
            self.table_cells.item(row, 1).setText(str(voxel_count))
            if len(centroid) == 3:
                centroid_str = f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"
            elif len(centroid) == 2:
                centroid_str = f"({centroid[0]:.1f}, {centroid[1]:.1f})"
            else:
                centroid_str = str(centroid)
            self.table_cells.item(row, 2).setText(centroid_str)
    
    def _on_table_scroll(self):
        """Handle table scroll to compute stats for newly visible cells."""
        self._compute_visible_cell_stats()
    
    def _on_cell_selection_changed(self):
        """Handle table selection changes."""
        selected_rows = self.table_cells.selectionModel().selectedRows()
        self._selected_cell_ids = {
            self._cell_ids[row.row()] 
            for row in selected_rows
            if row.row() < len(self._cell_ids)
        }
        self._update_cell_count_label()
    
    def _select_all_cells(self):
        """Select all cells in the table."""
        self.table_cells.selectAll()
        self._on_cell_selection_changed()
    
    def _clear_cell_selection(self):
        """Clear cell selection."""
        self.table_cells.clearSelection()
        self._selected_cell_ids.clear()
        self._update_cell_count_label()
    
    def _update_cell_count_label(self):
        """Update the cell count status label."""
        total = len(self._cell_ids)
        selected = len(self._selected_cell_ids)
        self.lbl_cell_count.setText(f"Selected: {selected} | Total: {total} cells")
    
    def _highlight_selected_cells(self):
        """Highlight selected cells in both 2D and 3D view."""
        mask_layer = self._get_mask_layer()
        if mask_layer is None:
            self.lbl_status.setText("No mask layer selected")
            self.lbl_status.setStyleSheet("color: orange;")
            return
        
        if not self._selected_cell_ids:
            self.lbl_status.setText("No cells selected")
            self.lbl_status.setStyleSheet("color: orange;")
            return
        
        # Clear any existing highlight first
        self._clear_cell_highlight()
        
        # Store original opacity for restoration
        self._original_mask_opacity = mask_layer.opacity
        
        # Create selection mask containing only selected cells
        selection_mask = np.zeros_like(mask_layer.data)
        for cell_id in self._selected_cell_ids:
            selection_mask[mask_layer.data == cell_id] = cell_id
        
        # Create overlay layer
        overlay_name = f"_cell_highlight_{mask_layer.name}"
        
        if selection_mask.max() > 0:
            overlay = self.viewer.add_labels(
                selection_mask,
                name=overlay_name,
                opacity=1.0,
                blending="translucent",
            )
            # Copy colormap from original
            overlay.colormap = mask_layer.colormap
            
            # Enable bounding box for 3D visualization
            overlay.bounding_box.visible = True
            overlay.bounding_box.line_color = "yellow"
            
            # Dim original mask
            mask_layer.opacity = 0.2
            
            self._cell_selection_overlay = overlay_name
            
            self.lbl_status.setText(f"Highlighting {len(self._selected_cell_ids)} cells")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
    
    def _clear_cell_highlight(self):
        """Remove cell highlight overlay and restore original appearance."""
        # Remove overlay layer if exists
        if self._cell_selection_overlay:
            for layer in list(self.viewer.layers):
                if layer.name == self._cell_selection_overlay:
                    self.viewer.layers.remove(layer)
                    break
            self._cell_selection_overlay = None
        
        # Restore original mask opacity
        mask_layer = self._get_mask_layer()
        if mask_layer is not None:
            mask_layer.opacity = self._original_mask_opacity
        
        self.lbl_status.setText("Highlight cleared")
        self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
    
    def _refresh_grid_mode(self):
        """Refresh grid mode after layer modifications to fix display issues.
        
        After modifying layer data or colors, napari's grid mode can get
        confused about layer positioning. This forces a refresh.
        
        Uses debouncing to prevent race conditions when multiple refreshes
        are requested in quick succession.
        """
        if not hasattr(self.viewer, 'grid') or not self.viewer.grid.enabled:
            return
        
        # Cancel any pending refresh to prevent race conditions
        if self._grid_refresh_timer is not None:
            self._grid_refresh_timer.stop()
            self._grid_refresh_timer.deleteLater()
        
        # Create new timer for debounced refresh
        self._grid_refresh_timer = QTimer()
        self._grid_refresh_timer.setSingleShot(True)
        self._grid_refresh_timer.timeout.connect(self._do_grid_refresh)
        self._grid_refresh_timer.start(100)  # 100ms delay for debouncing
    
    def _do_grid_refresh(self):
        """Actually perform the grid refresh.
        
        After modifying layer data, napari's grid mode can get confused about
        layer positioning. This method recalculates the grid shape based on
        visible layers (not all layers) and forces a complete refresh.
        
        The key insight is that napari's grid uses layer indices to assign
        grid positions, but hidden layers still "occupy" positions conceptually.
        We must recalculate the shape based only on visible layers.
        """
        # Clear timer reference
        self._grid_refresh_timer = None
        
        if not hasattr(self.viewer, 'grid') or not self.viewer.grid.enabled:
            return
        
        # Process pending events to ensure layer updates are complete
        QApplication.processEvents()
        
        # Count visible layers (same logic as group_widget)
        visible_count = sum(1 for layer in self.viewer.layers if layer.visible)
        
        if visible_count == 0:
            return
        
        # Calculate optimal grid shape for visible layers
        # Try to make it roughly square
        import math
        cols = math.ceil(math.sqrt(visible_count))
        rows = math.ceil(visible_count / cols)
        
        # Store current stride (we want to preserve this)
        current_stride = self.viewer.grid.stride
        
        # Set grid shape for visible layer count
        self.viewer.grid.shape = (rows, cols)
        
        # Toggle grid to force complete refresh
        # Note: Don't call processEvents() between off/on to avoid race conditions
        self.viewer.grid.enabled = False
        self.viewer.grid.enabled = True
        
        # Restore shape and stride after toggle to ensure they're applied
        self.viewer.grid.shape = (rows, cols)
        self.viewer.grid.stride = current_stride
        
        # Reset the view to ensure proper rendering
        self.viewer.reset_view()