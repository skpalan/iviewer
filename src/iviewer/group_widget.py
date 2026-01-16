"""
Layer Groups Widget for napari

Provides folder/group functionality to organize layers:
- Create, rename, and delete groups
- Drag and drop layers into groups
- Show only layers in selected group
- Works with both normal and grid view
"""

import re
from typing import Dict, List, Optional, Set

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QInputDialog,
    QMessageBox,
    QAbstractItemView,
    QMenu,
    QCheckBox,
    QLabel,
    QFrame,
)
from qtpy.QtGui import QDragEnterEvent, QDropEvent

import napari


class LayerGroupsWidget(QWidget):
    """Widget for organizing napari layers into groups/folders.
    
    Features:
    - Create groups to organize layers
    - Drag layers from napari's layer list into groups
    - Click a group to show only its layers
    - "Show All" to display all layers
    
    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer instance.
    """
    
    # Signal emitted when group visibility changes
    group_visibility_changed = Signal(str, bool)
    
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # Data structure: group_name -> set of layer names
        self._groups: Dict[str, Set[str]] = {}
        
        # Track which layers are assigned to groups
        self._layer_to_group: Dict[str, str] = {}
        
        # Currently active group filter (None = show all)
        self._active_group: Optional[str] = None
        
        # Store original visibility states for restoration
        self._original_visibility: Dict[str, bool] = {}
        
        # Store original grid shape for restoration
        self._original_grid_shape: Optional[tuple] = None
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header
        header = QLabel("<b>Layer Groups</b>")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        
        self.btn_new_group = QPushButton("+ Group")
        self.btn_new_group.setToolTip("Create a new group")
        self.btn_new_group.clicked.connect(self._create_group)
        btn_layout.addWidget(self.btn_new_group)
        
        self.btn_delete_group = QPushButton("- Group")
        self.btn_delete_group.setToolTip("Delete selected group")
        self.btn_delete_group.clicked.connect(self._delete_group)
        btn_layout.addWidget(self.btn_delete_group)
        
        self.btn_show_all = QPushButton("Show All")
        self.btn_show_all.setToolTip("Show all layers (exit group filter)")
        self.btn_show_all.clicked.connect(self._show_all_layers)
        btn_layout.addWidget(self.btn_show_all)
        
        layout.addLayout(btn_layout)
        
        # Second button row for Tidy
        btn_layout2 = QHBoxLayout()
        btn_layout2.setSpacing(4)
        
        self.btn_tidy = QPushButton("🧹 Tidy Layers")
        self.btn_tidy.setToolTip(
            "Auto-group layers by Gel/brain pattern (e.g., Gel20251024_brain08).\n"
            "Can be pressed multiple times to add new layers to existing groups."
        )
        self.btn_tidy.clicked.connect(self._tidy_layers)
        btn_layout2.addWidget(self.btn_tidy)
        
        layout.addLayout(btn_layout2)
        
        # Tree widget for groups and layers
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Groups / Layers"])
        self.tree.setDragEnabled(False)
        self.tree.setAcceptDrops(True)
        self.tree.setDragDropMode(QAbstractItemView.DropOnly)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemClicked.connect(self._on_item_clicked)
        
        # Enable drop events
        self.tree.dragEnterEvent = self._tree_drag_enter
        self.tree.dragMoveEvent = self._tree_drag_move
        self.tree.dropEvent = self._tree_drop
        
        layout.addWidget(self.tree)
        
        # "Ungrouped" section shows layers not in any group
        self.chk_show_ungrouped = QCheckBox("Show ungrouped layers when filtering")
        self.chk_show_ungrouped.setChecked(False)
        self.chk_show_ungrouped.setToolTip(
            "When viewing a specific group, also show layers not in any group"
        )
        self.chk_show_ungrouped.stateChanged.connect(self._apply_group_filter)
        layout.addWidget(self.chk_show_ungrouped)
        
        # Status label
        self.lbl_status = QLabel("Drag layers here to organize")
        self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)
        
        # Instructions
        instructions = QLabel(
            "<small>• Double-click group to filter view<br>"
            "• Drag layers from layer list into groups<br>"
            "• Right-click for more options</small>"
        )
        instructions.setStyleSheet("color: gray;")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
    
    def _connect_signals(self):
        """Connect to napari viewer signals."""
        # Listen for layer additions/removals
        self.viewer.layers.events.inserted.connect(self._on_layer_added)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.layers.events.reordered.connect(self._refresh_tree)
    
    def _create_group(self):
        """Create a new group."""
        name, ok = QInputDialog.getText(
            self, "New Group", "Enter group name:"
        )
        if ok and name:
            if name in self._groups:
                QMessageBox.warning(
                    self, "Error", f"Group '{name}' already exists."
                )
                return
            
            self._groups[name] = set()
            self._add_group_to_tree(name)
            self._update_status()
    
    def _delete_group(self):
        """Delete the selected group."""
        item = self.tree.currentItem()
        if item is None:
            QMessageBox.information(
                self, "Info", "Please select a group to delete."
            )
            return
        
        # Check if it's a group (top-level item)
        if item.parent() is not None:
            QMessageBox.information(
                self, "Info", "Please select a group (not a layer) to delete."
            )
            return
        
        # Get group name from stored data (not display text which has emoji)
        item_data = item.data(0, Qt.UserRole)
        if item_data is None or item_data.get("type") != "group":
            QMessageBox.information(
                self, "Info", "Please select a valid group to delete."
            )
            return
        
        group_name = item_data["name"]
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete group '{group_name}'?\n\nLayers will be ungrouped, not deleted.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        
        if reply == QMessageBox.Yes:
            # Remove layers from this group
            for layer_name in list(self._groups.get(group_name, [])):
                if layer_name in self._layer_to_group:
                    del self._layer_to_group[layer_name]
            
            # Remove group
            if group_name in self._groups:
                del self._groups[group_name]
            
            # Clear active filter if this was the active group
            if self._active_group == group_name:
                self._active_group = None
                self._restore_all_visibility()
            
            self._refresh_tree()
            self._update_status()
    
    def _add_group_to_tree(self, group_name: str):
        """Add a group item to the tree."""
        group_item = QTreeWidgetItem([f"📁 {group_name}"])
        group_item.setData(0, Qt.UserRole, {"type": "group", "name": group_name})
        group_item.setExpanded(True)
        self.tree.addTopLevelItem(group_item)
        
        # Add existing layers in this group
        for layer_name in self._groups.get(group_name, []):
            self._add_layer_to_group_item(group_item, layer_name)
    
    def _add_layer_to_group_item(self, group_item: QTreeWidgetItem, layer_name: str):
        """Add a layer item under a group."""
        # Determine icon based on layer type
        layer = self._get_layer_by_name(layer_name)
        if layer is not None:
            layer_type = type(layer).__name__
            icon_map = {
                "Image": "🖼️",
                "Points": "📍",
                "Labels": "🏷️",
                "Shapes": "⬡",
                "Surface": "🔺",
                "Vectors": "➡️",
                "Tracks": "〰️",
            }
            icon = icon_map.get(layer_type, "◻️")
        else:
            icon = "◻️"
        
        layer_item = QTreeWidgetItem([f"{icon} {layer_name}"])
        layer_item.setData(0, Qt.UserRole, {"type": "layer", "name": layer_name})
        group_item.addChild(layer_item)
    
    def _refresh_tree(self, event=None):
        """Refresh the entire tree view."""
        self.tree.clear()
        
        for group_name in sorted(self._groups.keys()):
            self._add_group_to_tree(group_name)
        
        self._update_status()
    
    def _get_layer_by_name(self, name: str):
        """Get a layer by name from the viewer."""
        for layer in self.viewer.layers:
            if layer.name == name:
                return layer
        return None
    
    def _on_layer_added(self, event):
        """Handle layer added to viewer."""
        layer = event.value
        # Layer is not in any group initially
        self._update_status()
        
        # If we have an active filter, decide visibility
        if self._active_group is not None:
            layer.visible = False
    
    def _on_layer_removed(self, event):
        """Handle layer removed from viewer."""
        layer = event.value
        layer_name = layer.name
        
        # Remove from group if assigned
        if layer_name in self._layer_to_group:
            group_name = self._layer_to_group[layer_name]
            if group_name in self._groups:
                self._groups[group_name].discard(layer_name)
            del self._layer_to_group[layer_name]
        
        # Remove from original visibility tracking
        if layer_name in self._original_visibility:
            del self._original_visibility[layer_name]
        
        self._refresh_tree()
    
    def _tree_drag_enter(self, event: QDragEnterEvent):
        """Handle drag enter on tree."""
        # Accept drops from napari layer list
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.accept()
    
    def _tree_drag_move(self, event):
        """Handle drag move on tree."""
        event.accept()
    
    def _tree_drop(self, event: QDropEvent):
        """Handle drop on tree - supports multiple selected layers."""
        # Get the target item
        pos = event.pos()
        target_item = self.tree.itemAt(pos)
        
        if target_item is None:
            event.ignore()
            return
        
        # Determine target group
        item_data = target_item.data(0, Qt.UserRole)
        if item_data is None:
            event.ignore()
            return
        
        if item_data["type"] == "layer":
            # Dropped on a layer - use its parent group
            parent = target_item.parent()
            if parent is None:
                event.ignore()
                return
            item_data = parent.data(0, Qt.UserRole)
        
        if item_data["type"] != "group":
            event.ignore()
            return
        
        group_name = item_data["name"]
        
        # Collect all layers to add - from selection (supports multiple)
        layers_to_add = []
        
        # First, try to get all selected layers from viewer
        if self.viewer.layers.selection:
            layers_to_add = [layer.name for layer in self.viewer.layers.selection]
        
        # If no selection, try mime data for single layer
        if not layers_to_add:
            mime_data = event.mimeData()
            if mime_data.hasText():
                layer_name = mime_data.text()
                if layer_name:
                    layers_to_add = [layer_name]
        
        if layers_to_add:
            # Add all selected layers to the group
            for layer_name in layers_to_add:
                self._add_layer_to_group_silent(layer_name, group_name)
            
            # Refresh UI once after all additions
            self._refresh_tree()
            self._apply_group_filter()
            
            # Update status
            if len(layers_to_add) == 1:
                self.lbl_status.setText(f"Added '{layers_to_add[0]}' to '{group_name}'")
            else:
                self.lbl_status.setText(f"Added {len(layers_to_add)} layers to '{group_name}'")
            
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def _add_layer_to_group_silent(self, layer_name: str, group_name: str):
        """Add a layer to a group without UI updates (for batch operations)."""
        # Verify layer exists
        layer = self._get_layer_by_name(layer_name)
        if layer is None:
            return False
        
        # Remove from previous group if any
        if layer_name in self._layer_to_group:
            old_group = self._layer_to_group[layer_name]
            if old_group in self._groups:
                self._groups[old_group].discard(layer_name)
        
        # Add to new group
        if group_name not in self._groups:
            self._groups[group_name] = set()
        
        self._groups[group_name].add(layer_name)
        self._layer_to_group[layer_name] = group_name
        return True
    
    def _add_layer_to_group(self, layer_name: str, group_name: str):
        """Add a layer to a group."""
        # Verify layer exists
        layer = self._get_layer_by_name(layer_name)
        if layer is None:
            QMessageBox.warning(
                self, "Error", f"Layer '{layer_name}' not found."
            )
            return
        
        # Remove from previous group if any
        if layer_name in self._layer_to_group:
            old_group = self._layer_to_group[layer_name]
            if old_group in self._groups:
                self._groups[old_group].discard(layer_name)
        
        # Add to new group
        if group_name not in self._groups:
            self._groups[group_name] = set()
        
        self._groups[group_name].add(layer_name)
        self._layer_to_group[layer_name] = group_name
        
        self._refresh_tree()
        self._apply_group_filter()
        
        self.lbl_status.setText(f"Added '{layer_name}' to '{group_name}'")
    
    def _remove_layer_from_group(self, layer_name: str):
        """Remove a layer from its group."""
        if layer_name not in self._layer_to_group:
            return
        
        group_name = self._layer_to_group[layer_name]
        if group_name in self._groups:
            self._groups[group_name].discard(layer_name)
        del self._layer_to_group[layer_name]
        
        self._refresh_tree()
        self._apply_group_filter()
    
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle single click on tree item."""
        pass  # Reserved for future use
    
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on tree item - activate group filter."""
        item_data = item.data(0, Qt.UserRole)
        if item_data is None:
            return
        
        if item_data["type"] == "group":
            group_name = item_data["name"]
            self._activate_group_filter(group_name)
        elif item_data["type"] == "layer":
            # Double-click on layer - select it in viewer
            layer_name = item_data["name"]
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                self.viewer.layers.selection.clear()
                self.viewer.layers.selection.add(layer)
    
    def _activate_group_filter(self, group_name: str):
        """Activate filter to show only layers in the specified group."""
        # Store original visibility if not already stored
        if not self._original_visibility:
            for layer in self.viewer.layers:
                self._original_visibility[layer.name] = layer.visible
        
        # Store original grid shape if not already stored
        if self._original_grid_shape is None and hasattr(self.viewer, 'grid'):
            self._original_grid_shape = self.viewer.grid.shape
        
        self._active_group = group_name
        self._apply_group_filter()
        
        self.lbl_status.setText(f"Showing: {group_name}")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
    
    def _apply_group_filter(self, state=None):
        """Apply the current group filter to layer visibility."""
        if self._active_group is None:
            return
        
        group_layers = self._groups.get(self._active_group, set())
        show_ungrouped = self.chk_show_ungrouped.isChecked()
        
        for layer in self.viewer.layers:
            layer_name = layer.name
            in_active_group = layer_name in group_layers
            is_ungrouped = layer_name not in self._layer_to_group
            
            if in_active_group:
                layer.visible = True
            elif show_ungrouped and is_ungrouped:
                layer.visible = True
            else:
                layer.visible = False
        
        # Force grid mode refresh if active
        self._refresh_grid_mode()
    
    def _show_all_layers(self):
        """Show all layers (exit group filter mode)."""
        self._active_group = None
        self._restore_all_visibility()
        
        self.lbl_status.setText("Showing all layers")
        self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
    
    def _restore_all_visibility(self):
        """Restore original layer visibility and grid shape."""
        for layer in self.viewer.layers:
            if layer.name in self._original_visibility:
                layer.visible = self._original_visibility[layer.name]
            else:
                layer.visible = True
        
        self._original_visibility.clear()
        
        # Restore original grid shape if it was stored (with delay for proper refresh)
        if self._original_grid_shape is not None and hasattr(self.viewer, 'grid'):
            from qtpy.QtCore import QTimer
            QTimer.singleShot(50, self._do_restore_grid)
    
    def _do_restore_grid(self):
        """Actually restore the grid shape after visibility changes are processed."""
        if self._original_grid_shape is None or not hasattr(self.viewer, 'grid'):
            return
        
        # Process any pending Qt events
        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()
        
        self.viewer.grid.shape = self._original_grid_shape
        self._original_grid_shape = None
        
        # Toggle to refresh
        if self.viewer.grid.enabled:
            self.viewer.grid.enabled = False
            self.viewer.grid.enabled = True
    
    def _refresh_grid_mode(self):
        """Force refresh of grid mode to reflect visibility changes.
        
        Napari's grid mode calculates grid shape based on total layers,
        not visible ones. This method recalculates the grid shape to 
        match only the visible layers.
        """
        if not hasattr(self.viewer, 'grid') or not self.viewer.grid.enabled:
            return
        
        # Use a short delay to ensure visibility changes are processed first
        from qtpy.QtCore import QTimer
        QTimer.singleShot(50, self._do_grid_refresh)
    
    def _do_grid_refresh(self):
        """Actually perform the grid refresh after visibility changes are processed."""
        if not hasattr(self.viewer, 'grid') or not self.viewer.grid.enabled:
            return
        
        # Process any pending Qt events to ensure visibility is updated
        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Count visible layers
        visible_count = sum(1 for layer in self.viewer.layers if layer.visible)
        
        if visible_count == 0:
            return
        
        # Calculate optimal grid shape for visible layers
        # Try to make it roughly square
        import math
        cols = math.ceil(math.sqrt(visible_count))
        rows = math.ceil(visible_count / cols)
        
        # Store current stride
        current_stride = self.viewer.grid.stride
        
        # Update grid shape to match visible layer count
        self.viewer.grid.shape = (rows, cols)
        
        # Toggle grid to force refresh with new shape
        self.viewer.grid.enabled = False
        self.viewer.grid.enabled = True
        
        # Restore stride
        self.viewer.grid.stride = current_stride
    
    def _show_context_menu(self, position):
        """Show context menu for tree items."""
        item = self.tree.itemAt(position)
        if item is None:
            return
        
        item_data = item.data(0, Qt.UserRole)
        if item_data is None:
            return
        
        menu = QMenu(self)
        
        if item_data["type"] == "group":
            group_name = item_data["name"]
            
            action_filter = menu.addAction(f"Show only '{group_name}'")
            action_filter.triggered.connect(
                lambda: self._activate_group_filter(group_name)
            )
            
            action_rename = menu.addAction("Rename group")
            action_rename.triggered.connect(
                lambda: self._rename_group(group_name)
            )
            
            menu.addSeparator()
            
            action_delete = menu.addAction("Delete group")
            action_delete.triggered.connect(self._delete_group)
            
        elif item_data["type"] == "layer":
            layer_name = item_data["name"]
            
            action_select = menu.addAction("Select in viewer")
            action_select.triggered.connect(
                lambda: self._select_layer_in_viewer(layer_name)
            )
            
            action_remove = menu.addAction("Remove from group")
            action_remove.triggered.connect(
                lambda: self._remove_layer_from_group(layer_name)
            )
        
        menu.exec_(self.tree.viewport().mapToGlobal(position))
    
    def _rename_group(self, old_name: str):
        """Rename a group."""
        new_name, ok = QInputDialog.getText(
            self, "Rename Group", "Enter new name:", text=old_name
        )
        if ok and new_name and new_name != old_name:
            if new_name in self._groups:
                QMessageBox.warning(
                    self, "Error", f"Group '{new_name}' already exists."
                )
                return
            
            # Transfer layers to new group name
            self._groups[new_name] = self._groups.pop(old_name)
            
            # Update layer-to-group mapping
            for layer_name in self._groups[new_name]:
                self._layer_to_group[layer_name] = new_name
            
            # Update active filter if needed
            if self._active_group == old_name:
                self._active_group = new_name
            
            self._refresh_tree()
    
    def _select_layer_in_viewer(self, layer_name: str):
        """Select a layer in the napari viewer."""
        layer = self._get_layer_by_name(layer_name)
        if layer is not None:
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(layer)
    
    def _update_status(self):
        """Update the status label."""
        n_groups = len(self._groups)
        n_grouped = len(self._layer_to_group)
        n_total = len(self.viewer.layers)
        
        if self._active_group:
            return  # Don't override active group message
        
        if n_groups == 0:
            self.lbl_status.setText("No groups. Click '+ Group' to create one.")
        else:
            self.lbl_status.setText(
                f"{n_groups} group(s), {n_grouped}/{n_total} layers organized"
            )
        self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
    
    def add_selected_to_group(self, group_name: str):
        """Add currently selected layers to a group.
        
        This can be called programmatically or connected to a keyboard shortcut.
        
        Parameters
        ----------
        group_name : str
            Name of the group to add layers to.
        """
        if group_name not in self._groups:
            self._groups[group_name] = set()
        
        for layer in self.viewer.layers.selection:
            self._add_layer_to_group(layer.name, group_name)
    
    def get_group_layers(self, group_name: str) -> List[str]:
        """Get list of layer names in a group.
        
        Parameters
        ----------
        group_name : str
            Name of the group.
            
        Returns
        -------
        list of str
            Names of layers in the group.
        """
        return list(self._groups.get(group_name, []))
    
    def get_all_groups(self) -> List[str]:
        """Get list of all group names.
        
        Returns
        -------
        list of str
            Names of all groups.
        """
        return list(self._groups.keys())
    
    def _extract_group_key(self, layer_name: str) -> Optional[str]:
        """Extract group key from layer name based on Gel/brain pattern.
        
        Looks for patterns like:
        - Gel20251024_brain08
        - Gel20251024...brain08
        
        Parameters
        ----------
        layer_name : str
            The layer name to parse.
            
        Returns
        -------
        str or None
            Group key like 'Gel20251024_brain08', or None if pattern not found.
        """
        # Pattern to match Gel number (e.g., Gel20251024)
        gel_match = re.search(r'[Gg]el(\d+)', layer_name)
        # Pattern to match brain number (e.g., brain08)
        brain_match = re.search(r'[Bb]rain(\d+)', layer_name)
        
        if gel_match and brain_match:
            gel_num = gel_match.group(1)
            brain_num = brain_match.group(1)
            return f"Gel{gel_num}_brain{brain_num}"
        
        return None
    
    def _get_tidy_name(self, layer_name: str, group_key: str) -> str:
        """Generate a tidied/shortened layer name for display within a group.
        
        Removes redundant Gel/brain info since that's in the group name.
        Keeps channel names (Cy3, DAPI, GFP, etc.) and round info.
        
        Parameters
        ----------
        layer_name : str
            Original layer name.
        group_key : str
            The group key (e.g., 'Gel20251024_brain08').
            
        Returns
        -------
        str
            Shortened layer name.
        """
        # Extract key components from the layer name
        name = layer_name
        
        # Try to extract round and channel info
        round_match = re.search(r'[Rr]ound(\d+)', name)
        # Match channel- prefix or standalone channel names (DAPI, GFP, Cy3, Cy5, mCherry)
        channel_match = re.search(r'channel-([^_\s]+)', name, re.IGNORECASE)
        if not channel_match:
            # Try to match standalone channel names
            standalone_match = re.search(r'\b(DAPI|GFP|Cy3|Cy5|mCherry)\b', name, re.IGNORECASE)
            if standalone_match:
                channel_match = standalone_match
        
        parts = []
        
        # Check if it's a mask file - include Gel/brain prefix
        if 'cp_masks' in name.lower() or 'mask' in name.lower():
            parts.append(f"{group_key}_Masks")
        elif round_match:
            # Use full 'round' instead of 'R'
            parts.append(f"round{round_match.group(1)}")
        
        if channel_match:
            # group(1) works for both channel- prefix and standalone patterns
            channel = channel_match.group(1).replace(' Nar', '').replace('Nar', '')
            # Normalize case for common channels
            channel_upper = channel.upper()
            if channel_upper in ('DAPI', 'GFP'):
                channel = channel_upper
            parts.append(channel)
        
        # If we have meaningful parts, use them
        if parts:
            return '_'.join(parts)
        
        # Fallback: simplify by removing only Gel/brain patterns (keep channel names)
        simplified = re.sub(r'[Gg]el\d+_?', '', name)
        simplified = re.sub(r'[Bb]rain\d+_?', '', simplified)
        simplified = re.sub(r'^_+|_+$', '', simplified)  # Remove leading/trailing underscores
        simplified = re.sub(r'_+', '_', simplified)  # Collapse multiple underscores
        
        return simplified if simplified else name
    
    def _tidy_layers(self):
        """Auto-group layers by Gel/brain pattern and tidy their names.
        
        This method:
        1. Scans all layers for Gel/brain patterns
        2. Creates groups for each unique pattern
        3. Adds layers to appropriate groups
        4. Can be run multiple times - adds to existing groups
        5. Handles duplicates by adding suffix
        """
        # Track what we've done for status message
        groups_created = 0
        layers_grouped = 0
        layers_renamed = 0
        
        # First pass: identify groups and collect layers for each
        group_candidates: Dict[str, List[str]] = {}
        
        for layer in self.viewer.layers:
            layer_name = layer.name
            
            # Skip if already in a group
            if layer_name in self._layer_to_group:
                continue
            
            # Extract group key
            group_key = self._extract_group_key(layer_name)
            
            if group_key:
                if group_key not in group_candidates:
                    group_candidates[group_key] = []
                group_candidates[group_key].append(layer_name)
        
        # Second pass: create groups and add layers
        for group_key, layer_names in group_candidates.items():
            # Create group if it doesn't exist
            if group_key not in self._groups:
                self._groups[group_key] = set()
                groups_created += 1
            
            # Add layers to group
            for layer_name in layer_names:
                # Check for existing layers with same tidy name
                tidy_name = self._get_tidy_name(layer_name, group_key)
                
                # Check if a layer with this tidy name already exists in the group
                existing_tidy_names = set()
                for existing_layer in self._groups[group_key]:
                    existing_tidy = self._get_tidy_name(existing_layer, group_key)
                    existing_tidy_names.add(existing_tidy)
                
                # If duplicate, add suffix
                final_tidy_name = tidy_name
                suffix = 2
                while final_tidy_name in existing_tidy_names:
                    final_tidy_name = f"{tidy_name}_{suffix}"
                    suffix += 1
                
                # Rename layer if the tidy name is different
                layer = self._get_layer_by_name(layer_name)
                if layer is not None and final_tidy_name != layer_name:
                    # Check if a layer with final_tidy_name already exists
                    existing_names = {l.name for l in self.viewer.layers}
                    if final_tidy_name not in existing_names:
                        layer.name = final_tidy_name
                        layers_renamed += 1
                        # Use new name for grouping
                        layer_name = final_tidy_name
                
                # Add to group (using _add_layer_to_group_silent to avoid multiple refreshes)
                self._add_layer_to_group_silent(layer_name, group_key)
                layers_grouped += 1
        
        # Refresh the tree view
        self._refresh_tree()
        
        # Update status
        if groups_created > 0 or layers_grouped > 0:
            status_parts = []
            if groups_created > 0:
                status_parts.append(f"{groups_created} group(s) created")
            if layers_grouped > 0:
                status_parts.append(f"{layers_grouped} layer(s) grouped")
            if layers_renamed > 0:
                status_parts.append(f"{layers_renamed} renamed")
            self.lbl_status.setText(", ".join(status_parts))
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_status.setText("No layers to tidy (already grouped or no Gel/brain pattern)")
            self.lbl_status.setStyleSheet("color: gray; font-style: italic;")