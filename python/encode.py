import nibabel as nib
import numpy as np
import json
import struct
import numpy.linalg as LA
import sys
import argparse
import os

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

def save_meta_dict(shape_xyz, spacing_mm, affine, data_min, data_max, mode, endianness):
    dimX, dimY, dimZ = shape_xyz
    meta = {
        "dim": [int(dimX), int(dimY), int(dimZ)],
        "spacing_mm": spacing_mm,
        "dtype": "float32",
        "intensity_range": [float(data_min), float(data_max)],
        "affine": affine.tolist(),
        "mode": mode,
        "endianness": endianness,
        "order": "x-fast,y-then,z-outer"
    }
    return meta

def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[OK] Wrote {path}")

##########################################################
# NORMALISATION / MODES
##########################################################

def normalize_intensity_percentile(vol):
    p1, p99 = np.percentile(vol, [1, 99])
    vol_n = (vol - p1) / (p99 - p1)
    vol_n = np.clip(vol_n, 0, 1)
    return vol_n.astype(np.float32, copy=False), float(p1), float(p99)

##########################################################
# TRANSFER FUNCTIONS
##########################################################

def build_transfer_function_continuous(vol_norm, p1, p99, user_tf=None):
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

def build_transfer_function_labelmap(seg_data, user_tf=None):
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
        if user_tf and "labels" in user_tf and str(lbl) in user_tf["labels"]:
            spec = user_tf["labels"][str(lbl)]
            color = spec.get("color",[1.0,1.0,1.0])
            alpha = spec.get("alpha",0.5)
            name  = spec.get("name", f"label_{lbl}")
        else:
            rgba = default_rgba_for_label(int(lbl))
            color = [rgba[0], rgba[1], rgba[2]]
            alpha = rgba[3]
            name  = f"label_{lbl}"

        entries.append({
            "label": float(lbl),
            "color": [float(color[0]), float(color[1]), float(color[2])],
            "alpha": float(alpha),
            "name":  name
        })

    return {
        "type": "labelmap",
        "entries": entries
    }

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
            "alpha": float(alpha),
            "name": f"cluster_{cid}"
        })

    tf_json = {
        "type": "labelmap",
        "entries": cluster_rgba_entries,
        "origin": "auto_overlay"
    }

    return labelmap_3d.astype(np.float32), tf_json

def fuse_multichannel_to_labelmap(vol4d, user_tf=None):
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

    # background override
    bg_name  = "background"
    bg_color = [0.0,0.0,0.0]
    bg_alpha = 0.0
    if user_tf and "channels" in user_tf and "0" in user_tf["channels"]:
        spec0 = user_tf["channels"]["0"]
        bg_name  = spec0.get("name", bg_name)
        bg_color = spec0.get("color", bg_color)
        bg_alpha = spec0.get("alpha", bg_alpha)

    entries = [{
        "label": 0.0,
        "color": [float(bg_color[0]), float(bg_color[1]), float(bg_color[2])],
        "alpha": float(bg_alpha),
        "name":  bg_name
    }]

    channel_names_hint = []
    for cid in range(L):
        color = base_colors[cid % len(base_colors)]
        alpha = 0.5
        cname = f"channel_{cid}"

        if user_tf and "channels" in user_tf and str(cid+1) in user_tf["channels"]:
            spec = user_tf["channels"][str(cid+1)]
            color = spec.get("color", list(color))
            alpha = spec.get("alpha", alpha)
            cname = spec.get("name", cname)

        entries.append({
            "label": float(cid+1),
            "color": [float(color[0]), float(color[1]), float(color[2])],
            "alpha": float(alpha),
            "name":  cname
        })
        channel_names_hint.append(cname)

    tf_json = {
        "type": "labelmap",
        "entries": entries,
        "origin": "multi_label_channels",
        "channel_names_hint": channel_names_hint
    }

    return labelmap, tf_json

##########################################################
# VRDF WRITER
##########################################################

def write_vrdf(vrdf_path, meta_dict, tf_dict, volume_array):
    """
    Ecrit le conteneur .vrdf :
    [8 bytes magic ascii 'VRDF0001']
    [8 bytes uint64 total_size_bytes]

    Puis pour chaque bloc:
      [8 bytes uint64 block_size_meta_json]
      [meta_json_bytes...]
      [8 bytes uint64 block_size_tf_json]
      [tf_json_bytes...]
      [8 bytes uint64 block_size_raw]
      [raw_bytes...]

    Tous les entiers encodés en little-endian.
    """

    meta_bytes = json.dumps(meta_dict, separators=(",",":")).encode("utf-8")
    tf_bytes   = json.dumps(tf_dict, separators=(",",":")).encode("utf-8")

    X,Y,Z = volume_array.shape
    raw_buf = bytearray()
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                v = float(volume_array[x,y,z])
                raw_buf += struct.pack("<f", v)
    raw_bytes = bytes(raw_buf)

    block_meta_len = len(meta_bytes)
    block_tf_len   = len(tf_bytes)
    block_raw_len  = len(raw_bytes)

    total_size = (
        16 +
        8 + block_meta_len +
        8 + block_tf_len +
        8 + block_raw_len
    )

    with open(vrdf_path, "wb") as f:
        # magic + version
        f.write(b"VRDF0001")                          # 8 bytes
        f.write(struct.pack("<Q", total_size))        # 8 bytes uint64

        # META block
        f.write(struct.pack("<Q", block_meta_len))    # length meta
        f.write(meta_bytes)

        # TF block
        f.write(struct.pack("<Q", block_tf_len))      # length tf
        f.write(tf_bytes)

        # RAW block
        f.write(struct.pack("<Q", block_raw_len))     # length raw
        f.write(raw_bytes)

    print(f"[OK] Wrote {vrdf_path}")
    print(f"     total_size={total_size} bytes")
    print(f"     meta={block_meta_len} bytes, tf={block_tf_len} bytes, raw={block_raw_len} bytes")

##########################################################
# PIPELINE
##########################################################

def run_pipeline(
    nifti_path,
    mode,
    vrdf_out,
    auto_overlay_num_clusters,
    auto_overlay_min_alpha,
    user_cfg,
    debug_dump,
    raw_out,
    meta_out,
    tf_out
):
    img = nib.load(nifti_path)
    vol_full = img.get_fdata(dtype=np.float32)
    affine = img.affine
    spacing_mm = compute_spacing_mm(affine)

    print(f"[INFO] Original volume shape: {vol_full.shape}")
    print(f"[INFO] MODE: {mode}")

    user_tf = None
    if user_cfg is not None and "transfer_function" in user_cfg:
        user_tf = user_cfg["transfer_function"]

    if mode == "multi_label_channels":
        if vol_full.ndim != 4:
            raise ValueError("multi_label_channels attend un volume 4D (X,Y,Z,L).")
        export_data, tf_json = fuse_multichannel_to_labelmap(vol_full, user_tf=user_tf)
        data_min = float(export_data.min())
        data_max = float(export_data.max())

    elif mode == "labelmap":
        if vol_full.ndim == 4:
            print("[WARN] 4D volume detected, keeping only channel 0 for labelmap.")
            vol = vol_full[..., 0]
        else:
            vol = vol_full
        export_data = vol.astype(np.float32, copy=False)
        tf_json = build_transfer_function_labelmap(export_data, user_tf=user_tf)
        data_min = float(export_data.min())
        data_max = float(export_data.max())

    elif mode == "continuous":
        if vol_full.ndim == 4:
            print("[WARN] 4D volume detected, keeping only channel 0 for continuous.")
            vol = vol_full[..., 0]
        else:
            vol = vol_full
        export_data, p1, p99 = normalize_intensity_percentile(vol)
        tf_json = build_transfer_function_continuous(export_data, p1, p99, user_tf=user_tf)
        data_min = float(export_data.min())
        data_max = float(export_data.max())

    elif mode == "auto_overlay":
        if not (vol_full.ndim == 4 and vol_full.shape[3] == 3):
            print("[WARN] auto_overlay demandé mais volume != (X,Y,Z,3). Fallback continuous.")
            if vol_full.ndim == 4:
                vol = vol_full[...,0]
            else:
                vol = vol_full
            export_data, p1, p99 = normalize_intensity_percentile(vol)
            tf_json = build_transfer_function_continuous(export_data, p1, p99, user_tf=user_tf)
            data_min = float(export_data.min())
            data_max = float(export_data.max())
        else:
            export_data, tf_json = cluster_overlay_to_labelmap(
                vol_full,
                n_clusters=auto_overlay_num_clusters,
                alpha_default=auto_overlay_min_alpha
            )
            data_min = float(export_data.min())
            data_max = float(export_data.max())
    else:
        raise ValueError("MODE doit être 'labelmap', 'continuous', 'auto_overlay' ou 'multi_label_channels'.")

    meta_dict = save_meta_dict(
        export_data.shape,
        spacing_mm,
        affine,
        data_min,
        data_max,
        mode,
        sys.byteorder
    )

    write_vrdf(vrdf_out, meta_dict, tf_json, export_data)

    if debug_dump:
        write_raw_xyzC_order(export_data, raw_out)
        write_json(meta_out, meta_dict)
        write_json(tf_out, tf_json)

    print("----- SUMMARY -----")
    print("Shape:", export_data.shape)
    print("dtype:", export_data.dtype)
    print("min/max:", data_min, data_max)
    print("TF mode:", tf_json.get("type", "unknown"))
    print("TF origin:", tf_json.get("origin", "default"))
    if "entries" in tf_json:
        for e in tf_json["entries"]:
            lbl = e.get("label")
            nm  = e.get("name", None)
            if nm is not None:
                print(f"  label {lbl} -> {nm}")
    print("-------------------")

############################################
# CLI
############################################

def load_user_config(path):
    if path is None:
        return None
    with open(path, "r") as f:
        return json.load(f)

def parse_args():
    p = argparse.ArgumentParser(
        description="Convertit un NIfTI en .vrdf (volume+meta+TF). Optionnellement dump les fichiers séparés."
    )

    p.add_argument("--nifti", required=True, help="Chemin vers le .nii/.nii.gz d'entrée")
    p.add_argument("--mode", required=True,
                   choices=["labelmap","continuous","auto_overlay","multi_label_channels"],
                   help="Mode d'interprétation du volume")

    p.add_argument("--vrdf-out", default="scene.vrdf",
                   help="Chemin du fichier packagé .vrdf (défaut: scene.vrdf)")

    p.add_argument("--config", default=None,
                   help="Chemin vers un JSON de config utilisateur (mapping labels/canaux, couleurs, noms...)")

    p.add_argument("--num-clusters", type=int, default=4,
                   help="[auto_overlay] nombre de clusters KMeans")
    p.add_argument("--min-alpha", type=float, default=0.4,
                   help="[auto_overlay] alpha par défaut des clusters")

    p.add_argument("--debug-dump", action="store_true",
                   help="Si présent: écrit aussi volume.raw, volume_meta.json, transfer_function.json")

    p.add_argument("--raw-out", default="volume.raw",
                   help="[debug only] Chemin du .raw de sortie")
    p.add_argument("--meta-out", default="volume_meta.json",
                   help="[debug only] Chemin du metadata JSON")
    p.add_argument("--tf-out",   default="transfer_function.json",
                   help="[debug only] Chemin du transfer function JSON")

    args = p.parse_args()
    return args

def main():
    args = parse_args()
    user_cfg = load_user_config(args.config)

    run_pipeline(
        nifti_path=args.nifti,
        mode=args.mode,
        vrdf_out=args.vrdf_out,
        auto_overlay_num_clusters=args.num_clusters,
        auto_overlay_min_alpha=args.min_alpha,
        user_cfg=user_cfg,
        debug_dump=args.debug_dump,
        raw_out=args.raw_out,
        meta_out=args.meta_out,
        tf_out=args.tf_out
    )

if __name__ == "__main__":
    main()
