#!/usr/bin/env python3
"""
analyze.py
Usage:
    python analyze.py base_image.nii.gz [overlay_image.nii.gz]

- Displays NIfTI metadata
- Global voxel statistics
- Multi-channel analysis for 4D data (one channel per "class/probability")
- "Winner-takes-all" analysis (argmax) if 4D → simulates multi_label_channels mode
- Labelmap analysis if 3D discrete volume
- Mid-slices (axial/coronal/sagittal)
- Optional overlay visualization
"""

import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ------------------------
# Utilities
# ------------------------

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return img, data

def safe_filename(img):
    try:
        fm = getattr(img, "file_map", None)
        if fm is None:
            return "N/A"
        fileholder = fm.get("image", None)
        if fileholder is None:
            return "N/A"
        return getattr(fileholder, "filename", "N/A")
    except Exception:
        return "N/A"

def describe_nifti(img, data, name="Image"):
    header = img.header
    affine = img.affine

    print(f"=== {name} ===")
    print(f"File             : {safe_filename(img)}")
    print(f"NIfTI dtype      : {header.get_data_dtype()} (raw storage)")
    print(f"Shape            : {data.shape} (X, Y, Z[, C/T])")
    print(f"Num dimensions   : {data.ndim}")

    try:
        print(f"Voxel size (mm)  : {header.get_zooms()}")
    except Exception as e:
        print(f"Voxel size (mm)  : error -> {e}")

    print("Affine 4x4 (voxel -> mm):")
    print(affine)
    print()

    try:
        ornt = nib.aff2axcodes(affine)
        print(f"Axis orientation : {ornt} (X,Y,Z)")
    except Exception as e:
        print(f"Axis orientation : error -> {e}")
    print()

    finite_data = data[np.isfinite(data)]
    if finite_data.size > 0:
        print("Voxel statistics (finite values only):")
        print(f"  min   : {np.min(finite_data):.4f}")
        print(f"  max   : {np.max(finite_data):.4f}")
        print(f"  mean  : {np.mean(finite_data):.4f}")
        print(f"  std   : {np.std(finite_data):.4f}")
        print(f"  non-zero voxels : {np.count_nonzero(finite_data)} / {finite_data.size}")
    else:
        print("Voxel stats: empty image / NaN only")
    print()

    print("Header key fields:")
    keys_of_interest = [
        "descrip", "datatype", "bitpix",
        "scl_slope", "scl_inter",
        "qform_code", "sform_code",
        "xyzt_units", "toffset",
        "intent_name", "intent_code",
        "cal_min", "cal_max"
    ]
    for k in keys_of_interest:
        if k in header:
            print(f"  {k:12s}: {header[k]}")
    print("-" * 60)
    print()

def describe_multichannel(data, name="Image"):
    """
    If data is 4D (X,Y,Z,C), show per-channel stats and center voxel debug info
    to inspect class probabilities or activations.
    """
    if data.ndim != 4:
        return

    X, Y, Z, C = data.shape
    print(f"--- Multi-channel analysis for {name} ---")
    print(f"Detected 4D volume: {X}x{Y}x{Z} with {C} channels (last axis)")

    cx, cy, cz = X // 2, Y // 2, Z // 2
    center_vec = data[cx, cy, cz, :]

    print(f"Central voxel index = ({cx},{cy},{cz})")
    print("4D values at this voxel (all channels):")
    print("  [" + ", ".join([f"{v:.4f}" for v in center_vec]) + "]")
    print("Winning channel at center:", int(np.argmax(center_vec)), "(argmax)")
    print()

    for c in range(C):
        ch = data[..., c]
        finite_ch = ch[np.isfinite(ch)]
        if finite_ch.size == 0:
            print(f"[Channel {c}] (empty / NaN)")
            continue

        ch_min, ch_max = np.min(finite_ch), np.max(finite_ch)
        ch_mean, ch_std = np.mean(finite_ch), np.std(finite_ch)
        ch_nnz = np.count_nonzero(finite_ch)
        ch_total = finite_ch.size
        pct_nz = 100.0 * (ch_nnz / ch_total) if ch_total > 0 else 0.0

        print(f"[Channel {c}]")
        print(f"  min / max     : {ch_min:.4f} / {ch_max:.4f}")
        print(f"  mean / std    : {ch_mean:.4f} / {ch_std:.4f}")
        print(f"  non-zero vox. : {ch_nnz} / {ch_total}  (~{pct_nz:.2f}%)")
        print(f"  value @ center[{cx},{cy},{cz}] : {data[cx, cy, cz, c]:.4f}")

        sample = finite_ch[: min(finite_ch.size, 200000)]
        uniq_vals = np.unique(sample)

        if uniq_vals.size <= 6:
            vals_preview = ", ".join([f"{v:.3f}" for v in uniq_vals[:6]])
            print(f"  distinct values (sample): {vals_preview}")
        else:
            print(f"  distribution: seems continuous (not binary 0/1)")
        print()

    print("-" * 60)
    print()

def describe_multichannel_as_winner_map(data, name="Image"):
    """
    If data is 4D (X,Y,Z,C):
    - Compute, for each voxel, which channel wins (argmax)
    - Simulates multi_label_channels logic (without writing VRDF)
    - Show global histogram of winning labels.
    Useful to debug "everything is one class" issues.
    """
    if data.ndim != 4:
        return

    winner_channel = np.argmax(data, axis=3)
    winner_value   = np.max(data, axis=3)

    hard_labels = np.where(winner_value > 0,
                           winner_channel.astype(np.int32) + 1,
                           0).astype(np.int32)

    flat = hard_labels.reshape(-1)
    unique_lbls, counts = np.unique(flat, return_counts=True)
    total_vox = flat.size

    print(f"--- Winner-takes-all (argmax) analysis for {name} ---")
    print("Simulated labelmap (class distribution):")
    for lbl, cnt in zip(unique_lbls, counts):
        pct = 100.0 * cnt / total_vox if total_vox > 0 else 0.0
        print(f"  Label {lbl}: {cnt} voxels (~{pct:.3f}%)")
    print("Interpretation:")
    print("  - Label 0   = no active channel / background")
    print("  - Label 1   = channel 0 dominant")
    print("  - Label 2   = channel 1 dominant")
    print("  - Label 3   = channel 2 dominant")
    print("  - Label 4   = channel 3 dominant")
    print("-" * 60)
    print()

def describe_labelmap(data, name="Image"):
    """
    If data is 3D and has few unique values,
    show voxel count per label.
    This corresponds to an already argmaxed volume.
    """
    if data.ndim != 3:
        return

    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        return

    unique_vals = np.unique(finite_data)
    if unique_vals.size > 32:
        return

    total_vox = finite_data.size
    print(f"--- Labelmap analysis for {name} ---")
    for lbl in np.sort(unique_vals):
        count = int(np.sum(finite_data == lbl))
        pct = (100.0 * count / total_vox) if total_vox > 0 else 0.0
        print(f"  Label {lbl:.0f} : {count} voxels (~{pct:.3f}%)")
    print("-" * 60)
    print()

def get_middle_slices(data):
    """
    Returns the 3 central slices (sagittal / coronal / axial) and indices.
    Handles 4D volumes by using channel 0.
    """
    if data.ndim == 4:
        vol = data[..., 0]
    else:
        vol = data

    x_mid, y_mid, z_mid = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
    sag, cor, axi = np.rot90(vol[x_mid, :, :]), np.rot90(vol[:, y_mid, :]), np.rot90(vol[:, :, z_mid])

    return sag, cor, axi, (x_mid, y_mid, z_mid)

def show_preview(base_data, overlay_data=None, alpha=0.4, thr=None):
    """
    Attempts to display sagittal/coronal/axial views with optional overlay.
    On Windows without a GUI backend, plt.show() may fail → safely handled.
    """
    try:
        sag, cor, axi, mids = get_middle_slices(base_data)
        if overlay_data is not None:
            sag_o, cor_o, axi_o, _ = get_middle_slices(overlay_data)
        else:
            sag_o = cor_o = axi_o = None

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        def _draw(ax, base_slice, ov_slice, title):
            ax.imshow(base_slice, cmap="gray", origin="lower")
            if ov_slice is not None:
                if thr is not None:
                    tmp = ov_slice.copy()
                    tmp[tmp < thr] = np.nan
                    ax.imshow(tmp, cmap="hot", alpha=alpha, origin="lower")
                else:
                    ax.imshow(ov_slice, cmap="hot", alpha=alpha, origin="lower")
            ax.set_title(title)
            ax.axis("off")

        _draw(axes[0], sag, sag_o, f"Sagittal mid (x={mids[0]})")
        _draw(axes[1], cor, cor_o, f"Coronal  mid (y={mids[1]})")
        _draw(axes[2], axi, axi_o, f"Axial    mid (z={mids[2]})")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("⚠ Unable to display matplotlib preview (missing GUI backend).")
        print(f"  Reason: {e}")
        print("  → Text analysis above remains valid.")
        print("-" * 60)
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze.py base_image.nii.gz [overlay_image.nii.gz]")
        sys.exit(1)

    base_path = sys.argv[1]
    overlay_path = sys.argv[2] if len(sys.argv) > 2 else None

    base_img, base_data = load_nifti(base_path)
    describe_nifti(base_img, base_data, name="BASE IMAGE")
    describe_multichannel(base_data, name="BASE IMAGE")
    describe_multichannel_as_winner_map(base_data, name="BASE IMAGE")
    describe_labelmap(base_data, name="BASE IMAGE")

    overlay_img = None
    overlay_data = None

    if overlay_path is not None:
        overlay_img, overlay_data = load_nifti(overlay_path)
        describe_nifti(overlay_img, overlay_data, name="OVERLAY IMAGE")
        describe_multichannel(overlay_data, name="OVERLAY IMAGE")
        describe_multichannel_as_winner_map(overlay_data, name="OVERLAY IMAGE")
        describe_labelmap(overlay_data, name="OVERLAY IMAGE")

        same_shape = base_data.shape[:3] == overlay_data.shape[:3]
        same_affine = np.allclose(base_img.affine, overlay_img.affine, atol=1e-3)

        print("=== Overlay compatibility ===")
        print(f"Same shape (XYZ)? {same_shape} -> {base_data.shape[:3]} vs {overlay_data.shape[:3]}")
        print(f"Same affine (~mm space)? {same_affine}")
        if not same_affine:
            print("⚠ Warning: different affines → possible misalignment.")
        print("-" * 60)

    show_preview(base_data, overlay_data, alpha=0.4, thr=None)

if __name__ == "__main__":
    main()
