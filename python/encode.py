import nibabel as nib
import numpy as np
import json
import struct
import numpy.linalg as LA
import sys

############################################
# CONFIG UTILISATEUR
############################################

NIFTI_PATH = "brats_00012_separated-t2f.nii"
MODE = "multi_label_channels"  # "labelmap" | "continuous" | "auto_overlay" | "multi_label_channels"

RAW_OUT  = "volume.raw"
META_OUT = "volume_meta.json"
TF_OUT   = "transfer_function.json"

# for auto_overlay
AUTO_OVERLAY_NUM_CLUSTERS = 4
AUTO_OVERLAY_MIN_ALPHA    = 0.4

############################################
# FONCTIONS UTILITAIRES
############################################

def compute_spacing_mm(affine):
    sx = LA.norm(affine[0:3, 0])
    sy = LA.norm(affine[0:3, 1])
    sz = LA.norm(affine[0:3, 2])
    return [float(sx), float(sy), float(sz)]

def write_raw_xyzC_order(data, raw_path):
    dimX, dimY, dimZ = data.shape
    with open(raw_path, "wb") as f:
        for z in range(dimZ):
            for y in range(dimY):
                for x in range(dimX):
                    v = float(data[x, y, z])
                    f.write(struct.pack("<f", v))

def save_meta(meta_path, shape_xyz, spacing_mm, affine, data_min, data_max, mode):
    dimX, dimY, dimZ = shape_xyz
    meta = {
        "dim": [int(dimX), int(dimY), int(dimZ)],
        "spacing_mm": spacing_mm,
        "dtype": "float32",
        "intensity_range": [float(data_min), float(data_max)],
        "affine": affine.tolist(),
        "mode": mode,
        "endianness": sys.byteorder,
        "order": "x-fast,y-then,z-outer"
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Wrote {meta_path}")

##########################################################
# MODE "labelmap"
##########################################################

def build_transfer_function_labelmap(seg_data):
    labels_present = np.unique(seg_data).tolist()
    print("Labels présents:", labels_present)

    def default_rgba_for_label(lbl):
        if lbl == 0:
            return (0.0, 0.0, 0.0, 0.0)
        if lbl == 1:
            return (0.7, 0.0, 0.0, 0.4)
        if lbl == 2:
            return (0.3, 0.9, 0.3, 0.15)
        if lbl == 4:
            return (1.0, 0.5, 0.0, 0.6)

        rng = np.random.default_rng(int(lbl) % 123457)
        r,g,b = rng.uniform(0.3,1.0,3)
        a     = 0.4
        return (float(r), float(g), float(b), float(a))

    entries = []
    for lbl in labels_present:
        rgba = default_rgba_for_label(int(lbl))
        entries.append({
            "label": float(lbl),
            "color": [rgba[0], rgba[1], rgba[2]],
            "alpha": rgba[3]
        })
    return {
        "type": "labelmap",
        "entries": entries
    }

##########################################################
# MODE "continuous"
##########################################################

def normalize_intensity_percentile(vol):
    p1, p99 = np.percentile(vol, [1, 99])
    vol_n = (vol - p1) / (p99 - p1)
    vol_n = np.clip(vol_n, 0, 1)
    return vol_n.astype(np.float32, copy=False), float(p1), float(p99)

def build_transfer_function_continuous(vol_norm, p1, p99):
    tf_curve = []
    for i in range(256):
        x = i / 255.0

        if x < 0.25:
            t = x / 0.25
            r = 0.0
            g = 0.0 + 0.2*t
            b = 0.3 + 0.7*t
        elif x < 0.5:
            t = (x-0.25)/0.25
            r = 0.0 + 0.5*t
            g = 0.2 + 0.5*t
            b = 1.0 - 1.0*t
        elif x < 0.75:
            t = (x-0.5)/0.25
            r = 0.5 + 0.5*t
            g = 0.7 + 0.3*t
            b = 0.0
        else:
            t = (x-0.75)/0.25
            r = 1.0
            g = 1.0 - 0.5*t
            b = 0.0

        if x < 0.2:
            alpha = 0.0
        elif x < 0.6:
            alpha = 0.3 * (x-0.2)/(0.4)
        else:
            alpha = 0.3 + 0.7 * (x-0.6)/(0.4)

        tf_curve.append({
            "x": x,
            "color": [r,g,b],
            "alpha": alpha
        })

    return {
        "type": "continuous",
        "curve": tf_curve,
        "intensity_normalization": {
            "p1": p1,
            "p99": p99
        }
    }

##########################################################
# MODE "auto_overlay"
##########################################################

def cluster_overlay_to_labelmap(vol4d, n_clusters=4, alpha_default=0.4):
    from sklearn.cluster import KMeans

    X, Y, Z, C = vol4d.shape
    assert C == 3, "auto_overlay suppose (X,Y,Z,3)"

    flat_rgb = vol4d.reshape(-1, 3)

    rgb_min = flat_rgb.min(axis=0)
    rgb_max = flat_rgb.max(axis=0)
    rgb_range = np.maximum(rgb_max - rgb_min, 1e-6)
    flat_rgb_norm = (flat_rgb - rgb_min) / rgb_range

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(flat_rgb_norm)

    labelmap_3d = labels.reshape(X, Y, Z).astype(np.int32)

    cluster_rgba_entries = []
    for cid in range(n_clusters):
        mask = (labelmap_3d == cid).reshape(-1)
        if not np.any(mask):
            mean_col = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            alpha = 0.0
        else:
            mean_col = flat_rgb[mask].mean(axis=0).astype(np.float32)
            if mean_col.max() > 1.0:
                mean_col = mean_col / 255.0
            alpha = alpha_default

        cluster_rgba_entries.append({
            "label": float(cid),
            "color": [float(mean_col[0]), float(mean_col[1]), float(mean_col[2])],
            "alpha": float(alpha)
        })

    tf_json = {
        "type": "labelmap",
        "entries": cluster_rgba_entries,
        "origin": "auto_overlay"
    }

    return labelmap_3d.astype(np.float32), tf_json

##########################################################
# MODE "multi_label_channels"
##########################################################

def fuse_multichannel_to_labelmap(vol4d):
    """
    vol4d shape: (X,Y,Z,L) uint8 ou float
    Interprétation: chaque canal L = une classe.
    Sortie:
      labelmap (X,Y,Z) float32, avec:
        0 = rien
        1 = canal 0 actif
        2 = canal 1 actif
        3 = canal 2 actif
        ...
        L = canal L-1 actif
      tf_json de type labelmap avec couleurs fixes par canal.
    Règle en cas de chevauchement: on prend le canal avec l'intensité max.
    (si tie, np.argmax prend le premier)
    """

    X, Y, Z, L = vol4d.shape

    winner_channel = np.argmax(vol4d, axis=3)
    winner_value   = np.max(vol4d, axis=3)

    labelmap = np.where(
        winner_value > 0,
        winner_channel.astype(np.int32) + 1,
        0
    ).astype(np.float32)

    base_colors = [
        (1.0, 0.0, 0.0),   
        (0.0, 1.0, 0.0),   
        (0.0, 0.0, 1.0),   
        (1.0, 1.0, 0.0),   
        (1.0, 0.0, 1.0),   
        (0.0, 1.0, 1.0),   
        (1.0, 0.5, 0.0),   
        (0.6, 0.2, 0.8),
    ]

    entries = []
    entries.append({
        "label": 0.0,
        "color": [0.0, 0.0, 0.0],
        "alpha": 0.0
    })

    for cid in range(L):
        color = base_colors[cid % len(base_colors)]
        entries.append({
            "label": float(cid+1),
            "color": [color[0], color[1], color[2]],
            "alpha": 0.5
        })

    tf_json = {
        "type": "labelmap",
        "entries": entries,
        "origin": "multi_label_channels",
        "channel_names_hint": [
            f"channel_{i}" for i in range(L)
        ]
    }

    return labelmap, tf_json

##########################################################
# PIPELINE PRINCIPAL
##########################################################

def main():
    img = nib.load(NIFTI_PATH)
    vol_full = img.get_fdata(dtype=np.float32)
    affine = img.affine
    spacing_mm = compute_spacing_mm(affine)

    print(f"[INFO] Original volume shape: {vol_full.shape}")
    print(f"[INFO] MODE: {MODE}")

    if MODE == "multi_label_channels":
        #
        # Cas multi-couches (X,Y,Z,L)
        #
        if vol_full.ndim != 4:
            raise ValueError("multi_label_channels attend un volume 4D (X,Y,Z,L).")
        export_data, tf_json = fuse_multichannel_to_labelmap(vol_full)

        data_min = float(export_data.min())  # normalement 0
        data_max = float(export_data.max())  # normalement L

    elif MODE == "labelmap":
        #
        # Cas labelmap déjà fusionné voxel-wise (0,1,2,4,...)
        #
        if vol_full.ndim == 4:
            print("[WARN] 4D volume detected, keeping only channel 0 for labelmap.")
            vol = vol_full[..., 0]
        else:
            vol = vol_full
        export_data = vol.astype(np.float32, copy=False)
        tf_json = build_transfer_function_labelmap(export_data)
        data_min = float(export_data.min())
        data_max = float(export_data.max())

    elif MODE == "continuous":
        #
        # Cas intensité continue type IRM
        #
        if vol_full.ndim == 4:
            print("[WARN] 4D volume detected, keeping only channel 0 for continuous.")
            vol = vol_full[..., 0]
        else:
            vol = vol_full
        export_data, p1, p99 = normalize_intensity_percentile(vol)
        tf_json = build_transfer_function_continuous(export_data, p1, p99)
        data_min = float(export_data.min())
        data_max = float(export_data.max())

    elif MODE == "auto_overlay":
        #
        # Cas overlay RGB (X,Y,Z,3) qu'on clusterise
        #
        if not (vol_full.ndim == 4 and vol_full.shape[3] == 3):
            print("[WARN] auto_overlay demandé mais volume != (X,Y,Z,3). Fallback continuous.")
            if vol_full.ndim == 4:
                vol = vol_full[...,0]
            else:
                vol = vol_full
            export_data, p1, p99 = normalize_intensity_percentile(vol)
            tf_json = build_transfer_function_continuous(export_data, p1, p99)
            data_min = float(export_data.min())
            data_max = float(export_data.max())
        else:
            export_data, tf_json = cluster_overlay_to_labelmap(
                vol_full,
                n_clusters=AUTO_OVERLAY_NUM_CLUSTERS,
                alpha_default=AUTO_OVERLAY_MIN_ALPHA
            )
            data_min = float(export_data.min())
            data_max = float(export_data.max())
    else:
        raise ValueError("MODE doit être 'labelmap', 'continuous', 'auto_overlay' ou 'multi_label_channels'.")

    write_raw_xyzC_order(export_data, RAW_OUT)
    print(f"[OK] Wrote {RAW_OUT}")

    dimX, dimY, dimZ = export_data.shape
    save_meta(META_OUT,
              (dimX, dimY, dimZ),
              spacing_mm,
              affine,
              data_min,
              data_max,
              MODE)

    with open(TF_OUT, "w") as f:
        json.dump(tf_json, f, indent=2)
    print(f"[OK] Wrote {TF_OUT}")

    print("----- SUMMARY -----")
    print("Shape:", export_data.shape)
    print("dtype:", export_data.dtype)
    print("min/max:", data_min, data_max)
    print("TF mode:", tf_json.get("type", "unknown"))
    print("TF origin:", tf_json.get("origin", "default"))
    print("-------------------")

if __name__ == "__main__":
    main()
