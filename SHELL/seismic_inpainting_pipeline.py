"""
Seismic Inpainting Pipeline
----------------------------
Data & model center: kopal/MDA_GAN/MDA_GAN-main/
Workflow:
  1. Load a seismic image (2D or 3D .npy / .segy)
  2. Define a damaged region (rectangle)
  3. Reconstruct using the pretrained MDA_GAN model
  4. Show comparison: Original | Damaged | Reconstructed | Error
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F

# ============================================================================
# CONFIG — all paths derived from MDA_GAN as the single source of truth
# ============================================================================

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..", "MDA_GAN", "MDA_GAN-main")
DATA_DIR   = os.path.join(BASE_DIR, "data")
WEIGHTS    = os.path.join(BASE_DIR, "weights", "MDA_GAN.pt")

# ---- What to load ----
# Set INPUT_FILE to a path, or leave None to auto-pick the first .npy in DATA_DIR
INPUT_FILE = None

# ---- Damaged region (inline indices, depth indices) ----
# Set to None for auto: removes 10% of inline traces in the centre
DAMAGE_INLINE_START = None   # e.g. 50
DAMAGE_INLINE_END   = None   # e.g. 80
DAMAGE_DEPTH_START  = None   # None = full depth
DAMAGE_DEPTH_END    = None   # None = full depth

# ---- Which crossline slice to display (for 3D volumes) ----
DISPLAY_CROSSLINE = None     # None = middle

# ---- Crop volume before inference (speeds up CPU dramatically) ----
# Set to None to use full volume, or set a size e.g. 64
CROP_SIZE = 128   # uses volume[:64, :128, :128]

# ============================================================================
# SEISMIC COLORMAP
# ============================================================================

def seismic_cmap():
    return LinearSegmentedColormap.from_list(
        'seis', [(0.8, 0, 0), (1, 1, 1), (0, 0, 0.8)], N=256)

CMAP = seismic_cmap()

# ============================================================================
# LOAD DATA
# ============================================================================

def find_input_file():
    """Return the first .npy file in DATA_DIR."""
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {DATA_DIR}")
    return os.path.join(DATA_DIR, files[0])


def load_volume(path):
    """Load .npy or .segy and return a 3D numpy array (depth, inline, crossline)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        data = np.load(path)
    elif ext in (".segy", ".sgy"):
        try:
            import segyio
        except ImportError:
            raise ImportError("Install segyio: pip install segyio")
        with segyio.open(path, "r", ignore_geometry=True) as f:
            n_traces  = f.tracecount
            n_samples = f.samples.size
            data = np.zeros((n_traces, n_samples), dtype=np.float32)
            for i in range(n_traces):
                data[i] = f.trace[i]
        data = data.T                      # → (samples, traces) = (depth, inline)
        data = data[:, :, np.newaxis]      # → (depth, inline, 1)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # Ensure 3D
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    if data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    # Normalise to [-1, 1]
    mx = np.max(np.abs(data))
    if mx > 0:
        data = data / mx

    return data.astype(np.float32)

# ============================================================================
# BUILD DAMAGE MASK
# ============================================================================

def build_mask(volume, inline_start=None, inline_end=None,
               depth_start=None, depth_end=None):
    """
    Returns a boolean mask (depth, inline, crossline).
    True = damaged (to be reconstructed).
    Defaults: remove centre 10 % of inlines, full depth.
    """
    n_depth, n_inline, n_cross = volume.shape

    if inline_start is None:
        span = max(1, int(n_inline * 0.10))
        centre = n_inline // 2
        inline_start = centre - span // 2
        inline_end   = inline_start + span

    inline_end   = min(inline_end,   n_inline)
    depth_start  = depth_start  if depth_start  is not None else 0
    depth_end    = depth_end    if depth_end    is not None else n_depth

    mask = np.zeros(volume.shape, dtype=bool)
    mask[depth_start:depth_end, inline_start:inline_end, :] = True
    return mask, inline_start, inline_end, depth_start, depth_end

# ============================================================================
# MDA GAN INFERENCE
# ============================================================================

def run_mdagan(damaged_volume, device="cpu"):
    """
    Run the pretrained MDA GAN TorchScript model.
    Input:  (depth, inline, crossline) float32 in [-1, 1]
    Output: same shape, reconstructed values in [-1, 1]
    """
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")

    print(f"  Loading model from: {WEIGHTS}")
    model = torch.jit.load(WEIGHTS, map_location=device)
    model.eval()

    # Build 5-D tensor [1, 1, D, H, W]
    tensor = torch.from_numpy(damaged_volume).unsqueeze(0).unsqueeze(0)

    # Pad to multiples of 16 (U-Net requirement)
    _, _, d, h, w = tensor.shape
    pd = (16 - d % 16) % 16
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if pd or ph or pw:
        tensor = F.pad(tensor, (0, pw, 0, ph, 0, pd), mode="reflect")

    tensor = tensor.half()   # model weights are float16

    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.float()
        if pd or ph or pw:
            out = out[:, :, :d, :h, :w]

    return out.squeeze().numpy()

# ============================================================================
# RECONSTRUCTION
# ============================================================================

def reconstruct(volume, mask):
    """Apply damage then reconstruct with MDA GAN; return reconstructed volume."""
    damaged = volume.copy()
    damaged[mask] = 0.0

    print("\n[MDA GAN] Running inference...")
    gan_out = run_mdagan(damaged)

    result = volume.copy()
    result[mask] = gan_out[mask]
    return damaged, result

# ============================================================================
# METRICS
# ============================================================================

def metrics(original, reconstructed, mask):
    err   = original[mask] - reconstructed[mask]
    mse   = float(np.mean(err ** 2))
    psnr  = float(20 * np.log10(2.0 / (np.sqrt(mse) + 1e-12)))
    return mse, psnr

# ============================================================================
# VISUALISATION
# ============================================================================

def show_results(original, damaged, reconstructed, mask,
                 crossline_idx, inline_start, inline_end, mse, psnr, fname):

    slice_orig   = original    [:, :, crossline_idx]
    slice_dam    = damaged     [:, :, crossline_idx]
    slice_recon  = reconstructed[:, :, crossline_idx]
    mask_slice   = mask        [:, :, crossline_idx]

    error_slice  = np.zeros_like(slice_orig)
    error_slice[mask_slice] = np.abs(slice_orig - slice_recon)[mask_slice]

    vmax  = max(abs(slice_orig.min()), abs(slice_orig.max())) or 1.0
    emax  = error_slice.max() or 0.1

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"Seismic Inpainting  |  file: {os.path.basename(fname)}"
        f"  |  crossline {crossline_idx}"
        f"  |  MSE={mse:.5f}  PSNR={psnr:.2f} dB",
        fontsize=11, fontweight="bold"
    )

    titles  = ["Original", f"Damaged\n(inlines {inline_start}–{inline_end})",
               f"Reconstructed (MDA GAN)\nPSNR={psnr:.2f} dB", "Error map (masked)"]
    images  = [slice_orig, slice_dam, slice_recon, error_slice]
    cmaps   = [CMAP, CMAP, CMAP, "hot"]
    vmaxes  = [vmax, vmax, vmax, emax]
    vmins   = [-vmax, -vmax, -vmax, 0]
    colors  = ["yellow", "red", "lime", "cyan"]

    for ax, img, title, cm, vn, vx, col in zip(
            axes, images, titles, cmaps, vmins, vmaxes, colors):
        im = ax.imshow(img.T, cmap=cm, aspect="auto",
                       vmin=vn, vmax=vx, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Inline"); ax.set_ylabel("Depth")
        # Highlight damaged region
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (inline_start - 0.5, 0), inline_end - inline_start, img.shape[0],
            fill=False, edgecolor=col, linewidth=1.5, linestyle="--"
        )
        ax.add_patch(rect)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 68)
    print("  SEISMIC INPAINTING PIPELINE  —  powered by MDA GAN")
    print("=" * 68)

    # 1. Resolve input
    fpath = INPUT_FILE or find_input_file()
    print(f"\n[1] Loading: {fpath}")
    volume = load_volume(fpath)

    if CROP_SIZE is not None:
        volume = volume[:CROP_SIZE, :CROP_SIZE, :CROP_SIZE]
        print(f"    [crop] Using first {CROP_SIZE} samples per axis for speed")

    n_depth, n_inline, n_cross = volume.shape
    print(f"    Shape: depth={n_depth}  inline={n_inline}  crossline={n_cross}")

    # 2. Build damage mask
    print("\n[2] Building damage mask...")
    mask, il0, il1, d0, d1 = build_mask(
        volume,
        inline_start=DAMAGE_INLINE_START, inline_end=DAMAGE_INLINE_END,
        depth_start=DAMAGE_DEPTH_START,   depth_end=DAMAGE_DEPTH_END,
    )
    pct = 100 * mask.sum() / mask.size
    print(f"    Damaged inlines: {il0} → {il1}  ({il1-il0} traces)")
    print(f"    Damaged depth  : {d0} → {d1}")
    print(f"    Missing voxels : {mask.sum():,}  ({pct:.1f} %)")

    # 3. Reconstruct
    damaged, reconstructed = reconstruct(volume, mask)

    # 4. Metrics
    mse, psnr = metrics(volume, reconstructed, mask)
    print(f"\n[3] Results:  MSE = {mse:.6f}   PSNR = {psnr:.2f} dB")

    # 5. Display
    cx = DISPLAY_CROSSLINE if DISPLAY_CROSSLINE is not None else n_cross // 2
    show_results(volume, damaged, reconstructed, mask,
                 cx, il0, il1, mse, psnr, fpath)


if __name__ == "__main__":
    main()
