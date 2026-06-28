"""
SEISMIC INPAINTING GUI - 4 PANEL COMPARISON (SIMPLE WORKING VERSION)
No colorbar removal - just clear and redraw properly
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import traceback
import time

print("=" * 70)
print("SEISMIC INPAINTING - 4 PANEL COMPARISON")
print("=" * 70)

# Check PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("[OK] PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("[ERROR] PyTorch not available")

# Try to import segyio
try:
    import segyio
    SEGY_AVAILABLE = True
    print("[OK] segyio available")
except ImportError:
    SEGY_AVAILABLE = False
    print("[WARNING] segyio not available")

# ============================================================================
# SEISMIC COLORMAP (Red=Min, White=0, Blue=Max)
# ============================================================================

def create_seismic_colormap():
    colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]
    return LinearSegmentedColormap.from_list('seismic_custom', colors, N=256)

SEISMIC_CMAP = create_seismic_colormap()

# ============================================================================
# PATHS
# ============================================================================

mda_path = os.path.join(os.path.dirname(__file__), "..", "MDA_GAN", "MDA_GAN-main")
weights_file = os.path.join(mda_path, "weights", "MDA_GAN.pt")


# ============================================================================
# DETERMINISTIC METHOD (Linear Interpolation)
# ============================================================================

class DeterministicWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, volume_3d, mask_3d):
        super().__init__()
        self.volume_3d = volume_3d.copy()
        self.mask_3d = mask_3d.copy()
    
    def run(self):
        try:
            d, h, w = self.volume_3d.shape
            result = self.volume_3d.copy()
            
            mask_inline = np.any(self.mask_3d, axis=(0, 2))
            if np.any(mask_inline):
                inline_indices = np.where(mask_inline)[0]
                missing_start = inline_indices[0]
                missing_end = inline_indices[-1] + 1
            else:
                self.error.emit("No missing region found")
                return
            
            total_slices = d * w
            processed = 0
            
            for depth in range(d):
                for crossline in range(w):
                    mask_trace = self.mask_3d[depth, :, crossline]
                    missing_indices = np.where(mask_trace)[0]
                    
                    if len(missing_indices) > 0:
                        left = missing_start - 1
                        right = missing_end
                        
                        if left >= 0 and right < h:
                            for m in missing_indices:
                                weight_right = (m - left) / (right - left)
                                weight_left = 1 - weight_right
                                result[depth, m, crossline] = weight_left * self.volume_3d[depth, left, crossline] + weight_right * self.volume_3d[depth, right, crossline]
                    
                    processed += 1
                    if processed % 100 == 0:
                        self.progress.emit(int(100 * processed / total_slices))
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()


# ============================================================================
# STOCHASTIC METHOD (MDA GAN)
# ============================================================================

class StochasticWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, volume_3d, mask_3d):
        super().__init__()
        self.volume_3d = volume_3d.copy()
        self.mask_3d = mask_3d.copy()
    
    def run(self):
        try:
            self.status.emit("Loading MDA GAN model...")
            model = torch.jit.load(weights_file, map_location='cpu')
            model = model.float()   # force float32 — half precision freezes on CPU
            model.eval()

            self.status.emit("Preparing data...")
            damaged = self.volume_3d.copy()
            damaged[self.mask_3d] = 0
            volume_norm = damaged / (np.max(np.abs(damaged)) + 1e-8)

            n_d, n_h, n_w = volume_norm.shape

            # Smart crop: bounding box of mask + padding
            PAD = 16
            d_idx = np.where(np.any(self.mask_3d, axis=(1, 2)))[0]
            h_idx = np.where(np.any(self.mask_3d, axis=(0, 2)))[0]
            w_idx = np.where(np.any(self.mask_3d, axis=(0, 1)))[0]
            d0 = max(0,   d_idx[0]  - PAD);  d1 = min(n_d, d_idx[-1]  + PAD + 1)
            h0 = max(0,   h_idx[0]  - PAD);  h1 = min(n_h, h_idx[-1]  + PAD + 1)
            w0 = max(0,   w_idx[0]  - PAD);  w1 = min(n_w, w_idx[-1]  + PAD + 1)

            volume_crop = volume_norm[d0:d1, h0:h1, w0:w1]
            mask_crop   = self.mask_3d[d0:d1, h0:h1, w0:w1]

            self.status.emit(f"Cropped region: {volume_crop.shape} — running inference...")

            input_tensor = torch.from_numpy(volume_crop).float().unsqueeze(0).unsqueeze(0)
            _, _, d, h, w = input_tensor.shape
            pad_d = (16 - d % 16) % 16
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h, 0, pad_d), mode='reflect')

            self.status.emit("Running Stochastic (MDA GAN) inference...")
            
            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                output = output.float()
                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    output = output[:, :, :d, :h, :w]
                result = output.squeeze().numpy()
            
            data_max = np.max(np.abs(self.volume_3d))
            result = result * data_max
            final_result = self.volume_3d.copy()
            final_result[d0:d1, h0:h1, w0:w1][mask_crop] = result[mask_crop]
            
            self.progress.emit(100)
            self.finished.emit(final_result)
            
        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()


# ============================================================================
# 4-PANEL CANVAS (No colorbar removal issues)
# ============================================================================

class FourPanelCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(16, 10), dpi=100)
        super().__init__(self.figure)
        self.setParent(parent)
        self.parent_gui = parent

        # Reserve space: 2 rows x 2 cols of axes + colorbar cols beside each
        # GridSpec: 2 rows, 5 cols (plot, cbar, plot, cbar, tiny spare)
        gs = GridSpec(2, 5, figure=self.figure,
                      width_ratios=[1, 1, 1, 1, 0.06],
                      wspace=0.35, hspace=0.40)

        self.ax_orig    = self.figure.add_subplot(gs[0, 0])
        self.ax_damaged = self.figure.add_subplot(gs[0, 1])
        self.ax_recon   = self.figure.add_subplot(gs[1, 0])
        self.ax_error   = self.figure.add_subplot(gs[1, 1])

        # Colorbar axes (one per plot column pair — but we do one each)
        self.cax_orig    = self.figure.add_subplot(gs[0, 2])
        self.cax_damaged = self.figure.add_subplot(gs[0, 3])
        self.cax_recon   = self.figure.add_subplot(gs[1, 2])
        self.cax_error   = self.figure.add_subplot(gs[1, 3])

        # Hide the spare right strip axes (gs col 4 is unused visually)
        # actually we used cols 2&3 for cbars; hide col 4
        self._ax_spare = self.figure.add_subplot(gs[:, 4])
        self._ax_spare.set_visible(False)

        # Rectangle drawing
        self.rect_start = None
        self.rectangle = None
        self.drawing = False

        # Colorbars (created on first draw)
        self._cb_orig = None
        self._cb_damaged = None
        self._cb_recon = None
        self._cb_error = None

        # Connect mouse events
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)

    def _get_mask_bounds(self, mask):
        """Return (x1, x2, y1, y2) pixel bounds of the mask rectangle."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if np.any(rows) and np.any(cols):
            x1 = np.where(rows)[0][0]
            x2 = np.where(rows)[0][-1]
            y1 = np.where(cols)[0][0]
            y2 = np.where(cols)[0][-1]
            return x1, x2, y1, y2
        return None

    def update_panels(self, original, damaged, reconstructed, mask, method_name=""):
        """Update all 4 panels — always show full seismic field."""

        # Shared amplitude scale based on original
        vmax = max(abs(original.min()), abs(original.max()))
        if vmax == 0:
            vmax = 1.0

        # ---- Panel 1: ORIGINAL (full field) ----
        self.ax_orig.clear()
        self.cax_orig.clear()
        im1 = self.ax_orig.imshow(original.T, cmap=SEISMIC_CMAP, aspect='auto',
                                   vmin=-vmax, vmax=vmax, origin='lower')
        self.ax_orig.set_title('ORIGINAL\n(Draw rectangle here)', fontsize=10, fontweight='bold')
        self.ax_orig.set_xlabel('Inline', fontsize=9)
        self.ax_orig.set_ylabel('Depth', fontsize=9)
        if mask is not None and np.any(mask):
            b = self._get_mask_bounds(mask)
            if b:
                x1, x2, y1, y2 = b
                self.ax_orig.add_patch(
                    Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1 + 1, y2 - y1 + 1,
                               fill=False, edgecolor='yellow', linewidth=2))
        cb1 = self.figure.colorbar(im1, cax=self.cax_orig)
        cb1.set_label('Amplitude', fontsize=8)
        cb1.ax.tick_params(labelsize=7)

        # ---- Panel 2: DAMAGED (full field, missing zone zeroed) ----
        self.ax_damaged.clear()
        self.cax_damaged.clear()
        im2 = self.ax_damaged.imshow(damaged.T, cmap=SEISMIC_CMAP, aspect='auto',
                                      vmin=-vmax, vmax=vmax, origin='lower')
        self.ax_damaged.set_title('DAMAGED\n(Missing zone zeroed)', fontsize=10, fontweight='bold')
        self.ax_damaged.set_xlabel('Inline', fontsize=9)
        self.ax_damaged.set_ylabel('Depth', fontsize=9)
        if mask is not None and np.any(mask):
            b = self._get_mask_bounds(mask)
            if b:
                x1, x2, y1, y2 = b
                self.ax_damaged.add_patch(
                    Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1 + 1, y2 - y1 + 1,
                               fill=False, edgecolor='red', linewidth=2, linestyle='--'))
        cb2 = self.figure.colorbar(im2, cax=self.cax_damaged)
        cb2.set_label('Amplitude', fontsize=8)
        cb2.ax.tick_params(labelsize=7)

        # ---- Panel 3: RECONSTRUCTED (full field) ----
        self.ax_recon.clear()
        self.cax_recon.clear()
        im3 = self.ax_recon.imshow(reconstructed.T, cmap=SEISMIC_CMAP, aspect='auto',
                                    vmin=-vmax, vmax=vmax, origin='lower')
        self.ax_recon.set_title(f'RECONSTRUCTED\n({method_name})', fontsize=10, fontweight='bold')
        self.ax_recon.set_xlabel('Inline', fontsize=9)
        self.ax_recon.set_ylabel('Depth', fontsize=9)
        if mask is not None and np.any(mask):
            b = self._get_mask_bounds(mask)
            if b:
                x1, x2, y1, y2 = b
                self.ax_recon.add_patch(
                    Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1 + 1, y2 - y1 + 1,
                               fill=False, edgecolor='lime', linewidth=2, linestyle='--'))
        cb3 = self.figure.colorbar(im3, cax=self.cax_recon)
        cb3.set_label('Amplitude', fontsize=8)
        cb3.ax.tick_params(labelsize=7)

        # ---- Panel 4: ERROR MAP (full field — zero outside mask) ----
        self.ax_error.clear()
        self.cax_error.clear()

        if mask is not None and np.any(mask):
            error = np.abs(original - reconstructed)
            # Show error only inside mask; outside is 0 (dark on 'hot')
            error_display = np.zeros_like(error)
            error_display[mask] = error[mask]
            emax = error_display.max() if error_display.max() > 0 else 0.5
            im4 = self.ax_error.imshow(error_display.T, cmap='hot', aspect='auto',
                                        origin='lower', vmin=0, vmax=emax)
            self.ax_error.set_title('ERROR MAP\n(Abs error in missing zone)', fontsize=10, fontweight='bold')
            if mask is not None and np.any(mask):
                b = self._get_mask_bounds(mask)
                if b:
                    x1, x2, y1, y2 = b
                    self.ax_error.add_patch(
                        Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1 + 1, y2 - y1 + 1,
                                   fill=False, edgecolor='cyan', linewidth=1.5, linestyle=':'))
            mse = np.mean((original[mask] - reconstructed[mask]) ** 2)
            psnr = 20 * np.log10(2.0 / (np.sqrt(mse) + 1e-8)) if mse > 0 else float('inf')
            self.ax_error.text(0.02, 0.98, f'MSE: {mse:.6f}\nPSNR: {psnr:.2f} dB',
                               transform=self.ax_error.transAxes, fontsize=8,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        else:
            empty = np.zeros_like(original)
            im4 = self.ax_error.imshow(empty.T, cmap='hot', aspect='auto',
                                        origin='lower', vmin=0, vmax=0.5)
            self.ax_error.set_title('ERROR MAP\n(Draw rectangle first)', fontsize=10, fontweight='bold')

        self.ax_error.set_xlabel('Inline', fontsize=9)
        self.ax_error.set_ylabel('Depth', fontsize=9)
        cb4 = self.figure.colorbar(im4, cax=self.cax_error)
        cb4.set_label('Abs Error', fontsize=8)
        cb4.ax.tick_params(labelsize=7)

        self.draw()
    
    def on_press(self, event):
        if event.inaxes == self.ax_orig and event.button == 1:
            self.rect_start = (int(event.xdata), int(event.ydata))
            self.drawing = True
    
    def on_motion(self, event):
        if self.drawing and event.inaxes == self.ax_orig:
            x1, y1 = self.rect_start
            x2, y2 = int(event.xdata), int(event.ydata)
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            # redraw patches by clearing and re-adding
            for p in self.ax_orig.patches[:]:
                p.remove()
            self.rectangle = Rectangle((x, y), w, h, fill=False, edgecolor='yellow', linewidth=2)
            self.ax_orig.add_patch(self.rectangle)
            self.draw()
    
    def on_release(self, event):
        if self.drawing and event.inaxes == self.ax_orig:
            x1, y1 = self.rect_start
            x2, y2 = int(event.xdata), int(event.ydata)
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            self.drawing = False
            self.rectangle = None
            if w > 0 and h > 0 and self.parent_gui:
                self.parent_gui.create_mask(x, x+w, y, y+h)


# ============================================================================
# MAIN GUI
# ============================================================================

class SeismicInpaintingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.volume = None
        self.mask_3d = None
        self.result_volume = None
        self.current_slice = 0
        self.num_slices = 0
        self.original_slice = None
        self.damaged_slice = None
        self.init_ui()
        self.check_weights()
    
    def check_weights(self):
        if os.path.exists(weights_file):
            self.weights_label.setText("Weights: MDA_GAN.pt ✓")
        else:
            self.weights_label.setText("Weights: NOT FOUND")
    
    def init_ui(self):
        self.setWindowTitle("Seismic Inpainting - 4 Panel Comparison")
        self.setGeometry(100, 100, 1600, 1000)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # ==================== LEFT PANEL ====================
        left = QWidget()
        left.setMaximumWidth(300)
        left_layout = QVBoxLayout(left)
        
        # File loading
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        self.load_npy_btn = QPushButton("Load NPY File")
        self.load_npy_btn.clicked.connect(self.load_npy)
        file_layout.addWidget(self.load_npy_btn)
        self.load_segy_btn = QPushButton("Load SEG-Y File")
        self.load_segy_btn.clicked.connect(self.load_segy)
        file_layout.addWidget(self.load_segy_btn)
        self.file_label = QLabel("No file")
        file_layout.addWidget(self.file_label)
        left_layout.addWidget(file_group)
        
        # Info
        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("No data")
        info_layout.addWidget(self.info_label)
        left_layout.addWidget(info_group)
        
        # Weights
        weights_group = QGroupBox("Weights")
        weights_layout = QVBoxLayout(weights_group)
        self.weights_label = QLabel("Checking...")
        weights_layout.addWidget(self.weights_label)
        left_layout.addWidget(weights_group)
        
        # Slice control
        slice_group = QGroupBox("3D Slice")
        slice_layout = QVBoxLayout(slice_group)
        slice_row = QHBoxLayout()
        slice_row.addWidget(QLabel("Slice:"))
        self.slice_spin = QSpinBox()
        self.slice_spin.valueChanged.connect(self.change_slice)
        slice_row.addWidget(self.slice_spin)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.change_slice_slider)
        slice_row.addWidget(self.slice_slider)
        slice_layout.addLayout(slice_row)
        left_layout.addWidget(slice_group)
        slice_group.setVisible(False)
        
        # Mask
        mask_group = QGroupBox("Missing Region")
        mask_layout = QVBoxLayout(mask_group)
        self.clear_btn = QPushButton("Clear Mask")
        self.clear_btn.clicked.connect(self.clear_mask)
        mask_layout.addWidget(self.clear_btn)
        self.mask_label = QLabel("No mask - draw rectangle on ORIGINAL panel")
        mask_layout.addWidget(self.mask_label)
        left_layout.addWidget(mask_group)
        
        # Method
        method_group = QGroupBox("Method")
        method_layout = QVBoxLayout(method_group)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Deterministic (Linear Interpolation)", "Stochastic (MDA GAN)"])
        method_layout.addWidget(self.method_combo)
        self.run_btn = QPushButton("Run Inpainting")
        self.run_btn.clicked.connect(self.run_inpainting)
        self.run_btn.setEnabled(False)
        method_layout.addWidget(self.run_btn)
        left_layout.addWidget(method_group)
        
        # Progress
        self.progress = QProgressBar()
        left_layout.addWidget(self.progress)
        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)
        
        # Compare
        compare_group = QGroupBox("Compare")
        compare_layout = QVBoxLayout(compare_group)
        self.compare_btn = QPushButton("Show Before / After")
        self.compare_btn.clicked.connect(self.show_comparison)
        self.compare_btn.setEnabled(False)
        compare_layout.addWidget(self.compare_btn)
        left_layout.addWidget(compare_group)

        # Save
        save_group = QGroupBox("Save")
        save_layout = QVBoxLayout(save_group)
        self.save_btn = QPushButton("Save Result")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        save_layout.addWidget(self.save_btn)
        left_layout.addWidget(save_group)
        
        left_layout.addStretch()
        
        # ==================== RIGHT PANEL ====================
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.four_panel = FourPanelCanvas(self)
        right_layout.addWidget(self.four_panel)
        toolbar = NavigationToolbar(self.four_panel, self)
        right_layout.addWidget(toolbar)
        
        layout.addWidget(left)
        layout.addWidget(right)
        layout.setStretchFactor(left, 1)
        layout.setStretchFactor(right, 3)
    
    def load_npy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load NPY", "", "NPY Files (*.npy)")
        if path:
            try:
                data = np.load(path)
                self.process_data(data, path)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
    
    def load_segy(self):
        if not SEGY_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Install segyio: pip install segyio")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load SEG-Y", "", "SEG-Y Files (*.segy *.sgy)")
        if path:
            try:
                with segyio.open(path, "r", ignore_geometry=True) as f:
                    n_traces = f.tracecount
                    n_samples = f.samples.size
                    data = np.zeros((n_traces, n_samples))
                    for i in range(n_traces):
                        data[i] = f.trace[i]
                data = data / np.max(np.abs(data))
                self.process_data(data, path)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
    
    def process_data(self, data, path):
        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        if len(data.shape) != 3:
            QMessageBox.warning(self, "Error", f"Expected 3D, got {len(data.shape)}D")
            return
        
        self.volume = data / np.max(np.abs(data))
        self.mask_3d = None
        self.result_volume = None
        self.num_slices = self.volume.shape[2]
        self.current_slice = self.num_slices // 2
        
        self.info_label.setText(f"Shape: {self.volume.shape}")
        self.file_label.setText(os.path.basename(path))
        
        slice_group = self.findChild(QGroupBox, "3D Slice")
        if slice_group:
            slice_group.setVisible(True)
            self.slice_spin.setRange(0, self.num_slices - 1)
            self.slice_slider.setRange(0, self.num_slices - 1)
            self.slice_spin.setValue(self.current_slice)
            self.slice_slider.setValue(self.current_slice)
        
        self.run_btn.setEnabled(True)
        self.update_display()
    
    def change_slice(self, value):
        self.current_slice = value
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(value)
        self.slice_slider.blockSignals(False)
        self.update_display()
    
    def change_slice_slider(self, value):
        self.current_slice = value
        self.slice_spin.blockSignals(True)
        self.slice_spin.setValue(value)
        self.slice_spin.blockSignals(False)
        self.update_display()
    
    def update_display(self):
        if self.volume is None:
            return
        
        self.original_slice = self.volume[:, :, self.current_slice]
        
        if self.mask_3d is not None:
            mask_slice = self.mask_3d[:, :, self.current_slice]
            self.damaged_slice = self.original_slice.copy()
            self.damaged_slice[mask_slice] = 0
        else:
            self.damaged_slice = self.original_slice.copy()
            mask_slice = None
        
        if self.result_volume is not None:
            reconstructed = self.result_volume[:, :, self.current_slice]
        else:
            reconstructed = self.damaged_slice
        
        method = self.method_combo.currentText()
        self.four_panel.update_panels(self.original_slice, self.damaged_slice, reconstructed, mask_slice, method)
    
    def create_mask(self, x1, x2, y1, y2):
        if self.volume is None:
            return
        d, h, w = self.volume.shape
        self.mask_3d = np.zeros_like(self.volume, dtype=bool)
        self.mask_3d[:, x1:x2, y1:y2] = True
        missing = np.sum(self.mask_3d)
        percent = 100 * missing / self.mask_3d.size
        self.mask_label.setText(f"Mask: inline {x1}-{x2}, depth {y1}-{y2}\n{missing} voxels ({percent:.1f}%)")
        self.update_display()
    
    def clear_mask(self):
        self.mask_3d = None
        self.result_volume = None
        self.mask_label.setText("No mask - draw rectangle on ORIGINAL panel")
        self.save_btn.setEnabled(False)
        self.update_display()
    
    def run_inpainting(self):
        if self.volume is None or self.mask_3d is None:
            QMessageBox.warning(self, "Warning", "Load data and draw a rectangle first!")
            return
        
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        
        method = self.method_combo.currentText()
        if "Deterministic" in method:
            self.worker = DeterministicWorker(self.volume, self.mask_3d)
        else:
            if not os.path.exists(weights_file):
                QMessageBox.critical(self, "Error", f"Weights not found: {weights_file}")
                self.run_btn.setEnabled(True)
                return
            self.worker = StochasticWorker(self.volume, self.mask_3d)
        
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_finished(self, result):
        self.result_volume = result
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.compare_btn.setEnabled(True)
        # Jump to a crossline inside the mask region
        if self.mask_3d is not None:
            w_idx = np.where(np.any(self.mask_3d, axis=(0, 1)))[0]
            if len(w_idx) > 0:
                self.current_slice = int((w_idx[0] + w_idx[-1]) // 2)
        self.slice_spin.blockSignals(True)
        self.slice_slider.blockSignals(True)
        self.slice_spin.setValue(self.current_slice)
        self.slice_slider.setValue(self.current_slice)
        self.slice_spin.blockSignals(False)
        self.slice_slider.blockSignals(False)
        self.update_display()
        self.status_label.setText("Complete!")
    
    def on_error(self, msg):
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)
        self.status_label.setText("Error")
    
    def show_comparison(self):
        if self.volume is None or self.result_volume is None or self.mask_3d is None:
            return

        cx = self.current_slice
        orig  = self.volume      [:, :, cx]
        recon = self.result_volume[:, :, cx]
        mask  = self.mask_3d     [:, :, cx]

        # Find bounding box of mask in this slice
        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            QMessageBox.warning(self, "Compare", "No mask in current slice. Move the slice slider into the cropped region.")
            return

        r0, r1 = max(0, rows[0] - 10),  min(orig.shape[0], rows[-1] + 10)
        c0, c1 = max(0, cols[0] - 10),  min(orig.shape[1], cols[-1] + 10)

        orig_zoom  = orig [r0:r1, c0:c1]
        recon_zoom = recon[r0:r1, c0:c1]
        mask_zoom  = mask [r0:r1, c0:c1]
        error_zoom = np.abs(orig_zoom - recon_zoom)

        vmax = max(abs(orig_zoom.min()), abs(orig_zoom.max())) or 1.0
        emax = error_zoom[mask_zoom].max() if mask_zoom.any() else 0.1

        mse  = float(np.mean((orig_zoom[mask_zoom] - recon_zoom[mask_zoom])**2)) if mask_zoom.any() else 0
        psnr = float(20 * np.log10(2.0 / (np.sqrt(mse) + 1e-12))) if mse > 0 else 0

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(
            f"Before / After Comparison  —  crossline {cx}"
            f"  |  MSE={mse:.5f}  PSNR={psnr:.2f} dB",
            fontsize=12, fontweight="bold"
        )

        kw = dict(cmap=SEISMIC_CMAP, aspect="auto", origin="lower", vmin=-vmax, vmax=vmax)

        im0 = axes[0].imshow(orig_zoom.T,  **kw)
        axes[0].set_title("ORIGINAL\n(zoomed to mask region)", fontsize=11)
        axes[0].set_xlabel("Depth"); axes[0].set_ylabel("Inline")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(recon_zoom.T, **kw)
        axes[1].set_title(f"RECONSTRUCTED (MDA GAN)\nPSNR = {psnr:.2f} dB", fontsize=11)
        axes[1].set_xlabel("Depth")
        plt.colorbar(im1, ax=axes[1])

        err_display = np.zeros_like(error_zoom)
        err_display[mask_zoom] = error_zoom[mask_zoom]
        im2 = axes[2].imshow(err_display.T, cmap="hot", aspect="auto", origin="lower",
                              vmin=0, vmax=emax)
        axes[2].set_title("ERROR MAP\n(only inside masked region)", fontsize=11)
        axes[2].set_xlabel("Depth")
        plt.colorbar(im2, ax=axes[2])

        # Trace comparison below
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        mid_inline = (cols[0] + cols[-1]) // 2 - c0
        mid_inline = max(0, min(mid_inline, orig_zoom.shape[1] - 1))
        ax2.plot(orig_zoom [:, mid_inline], "b-",  linewidth=2, label="Original")
        ax2.plot(recon_zoom[:, mid_inline], "r--", linewidth=2, label="Reconstructed (MDA GAN)")
        ax2.axvspan(rows[0] - r0, rows[-1] - r0, alpha=0.15, color="orange", label="Masked region")
        ax2.set_title(f"Trace comparison — inline {mid_inline + c0}  |  crossline {cx}", fontsize=11)
        ax2.set_xlabel("Depth"); ax2.set_ylabel("Amplitude")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        fig2.tight_layout()

        fig.tight_layout()
        plt.show()

    def save_result(self):
        if self.result_volume is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Result", "", "NPY Files (*.npy)")
            if path:
                np.save(path, self.result_volume)
                QMessageBox.information(self, "Saved", path)


def main():
    app = QApplication(sys.argv)
    window = SeismicInpaintingGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()