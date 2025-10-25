#!/usr/bin/env python3
"""
analyze.py
Usage:
    python analyze.py base_image.nii.gz [overlay_image.nii.gz]

- Affiche métadonnées NIfTI
- Stats voxel
- Coupes milieu (axial/coronal/sagittal)
- Overlay optionnel
"""

import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ------------------------
# Utils
# ------------------------

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return img, data

def safe_filename(img):
    """
    Récupère le nom de fichier source depuis nibabel, compatible avec plusieurs versions.
    """
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
    print(f"Fichier          : {safe_filename(img)}")
    print(f"Type NIfTI       : {header.get_data_dtype()} (stockage brut)")
    print(f"Shape            : {data.shape} (X, Y, Z[, T])")
    print(f"Nb dimensions    : {data.ndim}")

    try:
        print(f"Voxel size (mm)  : {header.get_zooms()}")
    except Exception as e:
        print(f"Voxel size (mm)  : erreur -> {e}")

    print("Affine 4x4 (voxel -> mm) :")
    print(affine)
    print()

    try:
        ornt = nib.aff2axcodes(affine)
        print(f"Orientation axes : {ornt} (X,Y,Z)")
    except Exception as e:
        print(f"Orientation axes : erreur -> {e}")
    print()

    finite_data = data[np.isfinite(data)]
    if finite_data.size > 0:
        print("Stats voxels (valeurs finies seulement):")
        print(f"  min   : {np.min(finite_data):.4f}")
        print(f"  max   : {np.max(finite_data):.4f}")
        print(f"  mean  : {np.mean(finite_data):.4f}")
        print(f"  std   : {np.std(finite_data):.4f}")
        print(f"  voxels non-nuls : {np.count_nonzero(finite_data)} / {finite_data.size}")
    else:
        print("Stats voxels : image vide / NaN")
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

def get_middle_slices(data):
    """
    Renvoie les 3 coupes centrales (sagittale / coronale / axiale) + indices.
    Gère aussi le cas 4D (on prend volume 0).
    """
    if data.ndim == 4:
        vol = data[..., 0]
    else:
        vol = data

    x_mid = vol.shape[0] // 2
    y_mid = vol.shape[1] // 2
    z_mid = vol.shape[2] // 2

    sag = np.rot90(vol[x_mid, :, :])
    cor = np.rot90(vol[:, y_mid, :])
    axi = np.rot90(vol[:, :, z_mid])

    return sag, cor, axi, (x_mid, y_mid, z_mid)

def show_preview(base_data, overlay_data=None, alpha=0.4, thr=None):
    """
    Affiche 3 sous-graphiques : sagittal / coronal / axial.
    Superpose overlay si fourni.
    thr : seuil (valeurs < thr masquées)
    alpha : transparence overlay
    """
    sag, cor, axi, mids = get_middle_slices(base_data)

    if overlay_data is not None:
        sag_o, cor_o, axi_o, _ = get_middle_slices(overlay_data)
    else:
        sag_o = cor_o = axi_o = None

    fig, axes = plt.subplots(1, 3, figsize=(12,4))

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze.py base_image.nii.gz [overlay_image.nii.gz]")
        sys.exit(1)

    base_path = sys.argv[1]
    overlay_path = sys.argv[2] if len(sys.argv) > 2 else None

    base_img, base_data = load_nifti(base_path)
    describe_nifti(base_img, base_data, name="BASE IMAGE")

    overlay_img = None
    overlay_data = None

    if overlay_path is not None:
        overlay_img, overlay_data = load_nifti(overlay_path)
        describe_nifti(overlay_img, overlay_data, name="OVERLAY IMAGE")

        # Vérification compat
        same_shape = base_data.shape[:3] == overlay_data.shape[:3]
        same_affine = np.allclose(base_img.affine, overlay_img.affine, atol=1e-3)

        print("=== Compatibilité overlay ===")
        print(f"Same shape (XYZ) ? {same_shape} -> {base_data.shape[:3]} vs {overlay_data.shape[:3]}")
        print(f"Same affine (~mm space)? {same_affine}")
        if not same_affine:
            print("⚠ Attention: affines différentes -> potentielle mauvaise superposition spatiale.")
        print("-" * 60)

    show_preview(base_data, overlay_data, alpha=0.4, thr=None)

if __name__ == "__main__":
    main()
