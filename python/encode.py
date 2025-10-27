import nibabel as nib
import numpy as np
import json
import struct
import numpy.linalg as LA
import sys
import argparse
import os

############################################
# UTILITIES
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

def save_meta_dict(shape_xyz, spacing_mm, affine, data_min, data_max,
                   mode, endianness, extra=None):
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
    if extra is not None:
        # merge extra fields (channels, channel_names, etc.)
        for k, v in extra.items():
            meta[k] = v
    return meta

def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[OK] Wrote {path}")

##########################################################
# NORMALIZATION
##########################################################

def normalize_intensity_percentile(vol):
    p1, p99 = np.percentile(vol, [1, 99])
    if p99 - p1 < 1e-12:
        vol_n = np.zeros_like(vol, dtype=np.float32)
    else:
        vol_n = (vol - p1) / (p99 - p1)
        vol_n = np.clip(vol_n, 0, 1).astype(np.float32, copy=False)
    return vol_n, float(p1), float(p99)

def normalize_channel_sparse(ch_vol):
    """
    ch_vol: 3D float32 (often sparse)
    Returns:
      ch_norm [0..1],
      p1, p99
    """
    ch_vol = ch_vol.astype(np.float32, copy=False)
    mask = ch_vol > 0

    if np.any(mask):
        vals = ch_vol[mask]
        if vals.size > 50:
            p1, p99 = np.percentile(vals, [1, 99])
        else:
            p1 = np.min(vals)
            p99 = np.max(vals)

        if p99 - p1 < 1e-12:
            ch_norm = np.zeros_like(ch_vol, dtype=np.float32)
            ch_norm[mask] = 1.0
            return ch_norm, float(p1), float(p99)

        ch_norm = np.zeros_like(ch_vol, dtype=np.float32)
        ch_norm[mask] = (ch_vol[mask] - p1) / (p99 - p1)
        ch_norm = np.clip(ch_norm, 0, 1).astype(np.float32, copy=False)
        return ch_norm, float(p1), float(p99)
    else:
        return np.zeros_like(ch_vol, dtype=np.float32), 0.0, 1.0

##########################################################
# TRANSFER FUNCTIONS
##########################################################

def build_transfer_function_continuous(vol_norm, p1, p99, user_tf=None):
    """
    Continuous TF (MRI/CT-style, or monochrome overlay).
    """
    preset = "grayscale_clinical"
    if user_tf and "preset" in user_tf:
        preset = user_tf["preset"]

    curve = []
    for i in range(256):
        x = i / 255.0

        if preset == "grayscale_clinical":
            gray = x ** 0.8
            if x < 0.15:
                alpha = 0.0
            elif x < 0.7:
                alpha = 0.5 * (x - 0.15) / (0.7 - 0.15)
            else:
                alpha = 0.5 + 0.5 * (x - 0.7) / (0.3)
                if alpha > 1.0:
                    alpha = 1.0
            color = [gray, gray, gray]

        elif preset == "hot_edges":
            r = min(1.0, x * 3.0)
            g = min(1.0, max(0.0, (x - 0.33) * 3.0))
            b = min(1.0, max(0.0, (x - 0.66) * 3.0))
            alpha = x ** 1.5
            color = [r, g, b]

        else:  # spectrum_debug
            if x < 0.25:
                t = x / 0.25
                r = 0.0; g = t;   b = 1.0
            elif x < 0.5:
                t = (x - 0.25) / 0.25
                r = 0.0; g = 1.0; b = 1.0 - t
            elif x < 0.75:
                t = (x - 0.5) / 0.25
                r = t;   g = 1.0; b = 0.0
            else:
                t = (x - 0.75) / 0.25
                r = 1.0; g = 1.0 - 0.5 * t; b = 0.0

            if x < 0.2:
                alpha = 0.0
            else:
                alpha = 0.4 * (x - 0.2) / (0.8)
                if alpha > 0.6:
                    alpha = 0.6
            color = [r, g, b]

        curve.append({
            "x": x,
            "color": [float(color[0]), float(color[1]), float(color[2])],
            "alpha": float(alpha)
        })

    return {
        "type": "continuous",
        "curve": curve,
        "intensity_normalization": {
            "p1": p1,
            "p99": p99
        },
        "origin": "continuous_preset_" + preset
    }

def build_transfer_function_labelmap(seg_data, user_tf=None):
    """
    TF for discrete volumes.
    Includes 'name', 'color', and 'alpha' per label.
    """
    labels_present = np.unique(seg_data).tolist()
    print("Labels present:", labels_present)

    def default_rgba_for_label(lbl_int):
        if lbl_int == 0: return (0.0, 0.0, 0.0, 0.0)
        if lbl_int == 1: return (0.8, 0.8, 0.8, 0.1)
        if lbl_int == 2: return (0.0, 1.0, 0.0, 0.4)
        if lbl_int == 3: return (0.0, 0.0, 1.0, 0.5)
        if lbl_int == 4: return (1.0, 1.0, 0.0, 1.0)
        rng = np.random.default_rng(int(lbl_int) % 123457)
        r, g, b = rng.uniform(0.3, 1.0, 3)
        a = 0.4
        return (float(r), float(g), float(b), float(a))

    entries = []
    for lbl in labels_present:
        lbl_int = int(lbl)
        lbl_key = str(lbl_int)

        if user_tf and "labels" in user_tf and lbl_key in user_tf["labels"]:
            spec = user_tf["labels"][lbl_key]
            color = spec.get("color", [1.0, 1.0, 1.0])
            alpha = spec.get("alpha", 0.5)
            name  = spec.get("name", f"label_{lbl_int}")
        else:
            rgba = default_rgba_for_label(lbl_int)
            color = [rgba[0], rgba[1], rgba[2]]
            alpha = rgba[3]
            name  = f"label_{lbl_int}"

        entries.append({
            "label": float(lbl_int),
            "color": [float(color[0]), float(color[1]), float(color[2])],
            "alpha": float(alpha),
            "name":  name
        })

    return {
        "type": "labelmap",
        "entries": entries
    }

##########################################################
# VRDF WRITERS
##########################################################

def write_vrdf_like(vrdf_path, meta_dict, tf_dict, volume_array):
    """
    Write a single-channel VRDF (1 float32 per voxel).
    """
    meta_bytes = json.dumps(meta_dict, separators=(",",":")).encode("utf-8")
    tf_bytes   = json.dumps(tf_dict,   separators=(",",":")).encode("utf-8")

    X, Y, Z = volume_array.shape
    raw_buf = bytearray()
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                v = float(volume_array[x, y, z])
                raw_buf += struct.pack("<f", v)
    raw_bytes = bytes(raw_buf)

    block_meta_len = len(meta_bytes)
    block_tf_len   = len(tf_bytes)
    block_raw_len  = len(raw_bytes)

    total_size = 16 + (8 + block_meta_len) + (8 + block_tf_len) + (8 + block_raw_len)

    with open(vrdf_path, "wb") as f:
        f.write(b"VRDF0001")
        f.write(struct.pack("<Q", total_size))

        f.write(struct.pack("<Q", block_meta_len))
        f.write(meta_bytes)

        f.write(struct.pack("<Q", block_tf_len))
        f.write(tf_bytes)

        f.write(struct.pack("<Q", block_raw_len))
        f.write(raw_bytes)

    print(f"[OK] Wrote {vrdf_path}")
    print(f"     total_size={total_size} bytes")
    print(f"     meta={block_meta_len} bytes, tf={block_tf_len} bytes, raw={block_raw_len} bytes")

def write_vrdf_interleaved_label_weight(vrdf_path, meta_dict, tf_dict,
                                        labelmap, weights_norm):
    """
    Write a two-channel interleaved VRDF with [label, weight] per voxel.
    - labelmap: float32 (discrete class: 0,1,2,3..)
    - weights_norm: float32 [0..1]

    We interleave both channels per voxel (struct.pack two floats).
    Unity reads meta["channels"]=2 and splits them on load.
    """
    assert labelmap.shape == weights_norm.shape
    meta_bytes = json.dumps(meta_dict, separators=(",",":")).encode("utf-8")
    tf_bytes   = json.dumps(tf_dict,   separators=(",",":")).encode("utf-8")

    X, Y, Z = labelmap.shape
    raw_buf = bytearray()
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                v_label  = float(labelmap[x, y, z])
                v_weight = float(weights_norm[x, y, z])
                raw_buf += struct.pack("<f", v_label)
                raw_buf += struct.pack("<f", v_weight)
    raw_bytes = bytes(raw_buf)

    block_meta_len = len(meta_bytes)
    block_tf_len   = len(tf_bytes)
    block_raw_len  = len(raw_bytes)

    total_size = 16 + (8 + block_meta_len) + (8 + block_tf_len) + (8 + block_raw_len)

    with open(vrdf_path, "wb") as f:
        f.write(b"VRDF0001")
        f.write(struct.pack("<Q", total_size))

        f.write(struct.pack("<Q", block_meta_len))
        f.write(meta_bytes)

        f.write(struct.pack("<Q", block_tf_len))
        f.write(tf_bytes)

        f.write(struct.pack("<Q", block_raw_len))
        f.write(raw_bytes)

    print(f"[OK] Wrote {vrdf_path}")
    print(f"     total_size={total_size} bytes (interleaved label+weight)")
    print(f"     meta={block_meta_len} bytes, tf={block_tf_len} bytes, raw={block_raw_len} bytes")

##########################################################
# EXPORT MODES
##########################################################

def export_labelmap_case(vol_full, affine, spacing_mm, user_tf,
                         vrdf_out, debug_dump, raw_out, meta_out, tf_out):
    # if 4D, take channel 0
    if vol_full.ndim == 4:
        ch = 0
        print(f"[WARN] labelmap: 4D volume detected, using channel {ch} only.")
        vol = vol_full[..., ch]
    else:
        vol = vol_full

    export_data = vol.astype(np.float32, copy=False)
    tf_json = build_transfer_function_labelmap(export_data, user_tf=user_tf)

    data_min = float(export_data.min())
    data_max = float(export_data.max())

    meta_dict = save_meta_dict(
        export_data.shape,
        spacing_mm,
        affine,
        data_min,
        data_max,
        "labelmap",
        sys.byteorder
    )

    write_vrdf_like(vrdf_out, meta_dict, tf_json, export_data)

    if debug_dump:
        write_raw_xyzC_order(export_data, raw_out)
        write_json(meta_out, meta_dict)
        write_json(tf_out, tf_json)

    print("----- SUMMARY -----")
    print("Mode: labelmap")
    print("Shape:", export_data.shape)
    print("dtype:", export_data.dtype)
    print("min/max:", data_min, data_max)
    print("-------------------")

def export_continuous4d_case(vol_full, affine, spacing_mm, user_tf,
                             vrdf_out, debug_dump, raw_out, meta_out, tf_out):
    base_out = os.path.splitext(vrdf_out)[0]
    ext_out  = os.path.splitext(vrdf_out)[1]

    if vol_full.ndim == 3:
        vols = [vol_full]
    elif vol_full.ndim == 4:
        T = vol_full.shape[3]
        vols = [vol_full[..., t] for t in range(T)]
    else:
        raise ValueError("continuous4d expects 3D or 4D (X,Y,Z[,T]).")

    for t, vol3d in enumerate(vols):
        vol3d = vol3d.astype(np.float32, copy=False)
        vol_norm, p1, p99 = normalize_intensity_percentile(vol3d)

        tf_json = build_transfer_function_continuous(vol_norm, p1, p99, user_tf=user_tf)

        data_min = float(vol_norm.min())
        data_max = float(vol_norm.max())

        meta_dict = save_meta_dict(
            vol_norm.shape,
            spacing_mm,
            affine,
            data_min,
            data_max,
            "continuous",
            sys.byteorder
        )

        if len(vols) == 1:
            out_path = vrdf_out
            raw_dump_path = raw_out
            meta_dump_path = meta_out
            tf_dump_path = tf_out
        else:
            out_path = f"{base_out}_t{t:02d}{ext_out}"
            raw_dump_path = f"{base_out}_t{t:02d}.raw"
            meta_dump_path = f"{base_out}_t{t:02d}_meta.json"
            tf_dump_path   = f"{base_out}_t{t:02d}_tf.json"

        write_vrdf_like(out_path, meta_dict, tf_json, vol_norm)

        if debug_dump:
            write_raw_xyzC_order(vol_norm, raw_dump_path)
            write_json(meta_dump_path, meta_dict)
            write_json(tf_dump_path, tf_json)

        print("----- SUMMARY -----")
        print(f"Mode: continuous4d channel {t}")
        print("Out:", out_path)
        print("Shape:", vol_norm.shape)
        print("dtype:", vol_norm.dtype)
        print("min/max:", data_min, data_max)
        print("-------------------")

def export_labelmap_weighted_case(vol_full, affine, spacing_mm, user_tf,
                                  vrdf_out, debug_dump,
                                  split_weight=False):
    """
    labelmap_weighted4d
    1. computes:
       - discrete labelmap (argmax across channels)
       - weights_norm [0..1] (value of the winning channel)
    2. By default -> A SINGLE fused .vrdf:
         <base>_lw.vrdf
       - meta.mode = "anatomy_label_weighted"
       - meta.channels = 2
       - RAW block: [label, weight] per voxel (2 floats)
       - TF = labelmap TF (color comes from label, weight modulates alpha)
    3. Optional (--split-weight) -> 2 legacy files:
         <base>_labels.vrdf     (labelmap only)
         <base>_weights.vrdf    (weight [0..1] only, mode "activity_weight")
    """

    if vol_full.ndim != 4:
        raise ValueError("labelmap_weighted4d expects a 4D volume (X,Y,Z,C).")

    X, Y, Z, C = vol_full.shape
    print(f"[INFO] labelmap_weighted4d: 4D volume detected: {vol_full.shape}")

    winner_channel = np.argmax(vol_full, axis=3)
    winner_value   = np.max(vol_full, axis=3)

    channel_to_label = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
    }

    labelmap = np.vectorize(channel_to_label.get)(winner_channel).astype(np.float32)
    labelmap[winner_value <= 0] = 0.0

    # normalize weight
    weights_raw = winner_value.astype(np.float32)
    mask_pos = weights_raw > 0
    if np.any(mask_pos):
        p1, p99 = np.percentile(weights_raw[mask_pos], [1, 99])
    else:
        p1, p99 = (0.0, 1.0)

    if p99 - p1 < 1e-12:
        weights_norm = np.zeros_like(weights_raw, dtype=np.float32)
    else:
        weights_norm = (weights_raw - p1) / (p99 - p1)
        weights_norm = np.clip(weights_norm, 0, 1)

    weights_norm[labelmap == 0] = 0.0

    tf_label = build_transfer_function_labelmap(labelmap, user_tf=user_tf)

    base_noext, _ = os.path.splitext(vrdf_out)

    if split_weight:
        label_path  = base_noext + "_labels.vrdf"
        weight_path = base_noext + "_weights.vrdf"

        meta_label = save_meta_dict(
            labelmap.shape,
            spacing_mm,
            affine,
            float(labelmap.min()),
            float(labelmap.max()),
            "anatomy_label",
            sys.byteorder
        )

        meta_weight = save_meta_dict(
            weights_norm.shape,
            spacing_mm,
            affine,
            0.0,
            1.0,
            "activity_weight",
            sys.byteorder
        )
        tf_weight = {
            "type": "none",
            "description": "weight[0..1] used to modulate label alpha"
        }

        write_vrdf_like(label_path,  meta_label,  tf_label,  labelmap)
        write_vrdf_like(weight_path, meta_weight, tf_weight, weights_norm)

        if debug_dump:
            write_raw_xyzC_order(labelmap,   base_noext + "_labels.raw")
            write_json(base_noext + "_labels_meta.json",   meta_label)
            write_json(base_noext + "_labels_tf.json",     tf_label)

            write_raw_xyzC_order(weights_norm, base_noext + "_weights.raw")
            write_json(base_noext + "_weights_meta.json",  meta_weight)
            write_json(base_noext + "_weights_tf.json",    tf_weight)

        print("----- SUMMARY -----")
        print("Mode: labelmap_weighted4d (SPLIT)")
        print("Labels :", label_path)
        print("Weights:", weight_path)
        print("-------------------")

    else:
        fused_path = base_noext + "_lw.vrdf"

        meta_fused = save_meta_dict(
            labelmap.shape,
            spacing_mm,
            affine,
            0.0,
            1.0,
            "anatomy_label_weighted",
            sys.byteorder,
            extra={
                "channels": 2,
                "channel_meaning": ["labelmap", "weight01"]
            }
        )

        write_vrdf_interleaved_label_weight(
            fused_path,
            meta_fused,
            tf_label,
            labelmap,
            weights_norm
        )

        if debug_dump:
            write_raw_xyzC_order(labelmap,   base_noext + "_lw_label.raw")
            write_raw_xyzC_order(weights_norm, base_noext + "_lw_weight.raw")
            write_json(base_noext + "_lw_meta.json",   meta_fused)
            write_json(base_noext + "_lw_tf.json",     tf_label)

        print("----- SUMMARY -----")
        print("Mode: labelmap_weighted4d (FUSED)")
        print("Fused :", fused_path)
        print("Shape :", labelmap.shape, "(2 interleaved channels)")
        print("-------------------")

def export_multi_overlay4d_case(vol_full, affine, spacing_mm, user_tf,
                                vrdf_out, debug_dump):
    """
    multi_overlay4d
    - no argmax
    - each channel C of the 4D volume (X,Y,Z,C) becomes <base>_chX.vrdf
      (legacy .vrdfw no longer used)
    - TF: fixed color + alpha ∝ normalized intensity
    """
    if vol_full.ndim != 4:
        raise ValueError("multi_overlay4d expects a 4D volume (X,Y,Z,C).")

    base_noext, _ = os.path.splitext(vrdf_out)

    channel_defs = {
        0: {"name": "Enhancing Tumor", "rgba": [1.0, 0.0, 0.0, 0.4]},
        1: {"name": "Core Tumor",      "rgba": [0.0, 1.0, 0.0, 0.4]},
        2: {"name": "Edema",           "rgba": [0.0, 0.0, 1.0, 0.4]},
        3: {"name": "Necrosis",        "rgba": [1.0, 1.0, 0.0, 0.4]},
    }

    X, Y, Z, C = vol_full.shape
    print(f"[INFO] multi_overlay4d: 4D volume detected: {vol_full.shape}")

    for cid_check in range(C):
        ch_tmp = vol_full[..., cid_check].astype(np.float32, copy=False)
        nnz = np.count_nonzero(ch_tmp)
        print(f"[DEBUG] channel {cid_check}: "
              f"min={ch_tmp.min():.6f} max={ch_tmp.max():.6f} nnz={nnz} sum={float(ch_tmp.sum()):.6f}")

    for cid in range(C):
        ch_vol = vol_full[..., cid].astype(np.float32, copy=False)

        ch_norm, p1, p99 = normalize_channel_sparse(ch_vol)
        nnz_norm = np.count_nonzero(ch_norm)
        print(f"[DEBUG] EXPORT channel {cid}: "
              f"p1={p1:.6f}, p99={p99:.6f}, "
              f"norm_min={ch_norm.min():.6f}, norm_max={ch_norm.max():.6f}, "
              f"nnz_norm={nnz_norm}, sum_norm={float(ch_norm.sum()):.6f}")

        ch_def = channel_defs.get(cid, {
            "name": f"channel_{cid}",
            "rgba": [1.0, 1.0, 1.0, 0.4]
        })
        r_base, g_base, b_base, a_base = ch_def["rgba"]

        curve = []
        for i in range(256):
            x = i / 255.0
            curve.append({
                "x": x,
                "color": [float(r_base), float(g_base), float(b_base)],
                "alpha": float(a_base * x)
            })

        tf_json = {
            "type": "continuous",
            "curve": curve,
            "origin": f"multi_overlay4d_channel_{cid}"
        }

        data_min = float(ch_norm.min())
        data_max = float(ch_norm.max())

        meta_dict = save_meta_dict(
            ch_norm.shape,
            spacing_mm,
            affine,
            data_min,
            data_max,
            "overlay_channel",
            sys.byteorder
        )

        out_path = f"{base_noext}_ch{cid}.vrdf"
        write_vrdf_like(out_path, meta_dict, tf_json, ch_norm)

        if debug_dump:
            write_raw_xyzC_order(ch_norm,     f"{base_noext}_ch{cid}.raw")
            write_json(f"{base_noext}_ch{cid}_meta.json", meta_dict)
            write_json(f"{base_noext}_ch{cid}_tf.json",   tf_json)

        print(f"[OK] wrote overlay channel {cid} -> {out_path} ({ch_def['name']})")

##########################################################
# CLI
##########################################################

def load_user_config(path):
    if path is None:
        return None
    with open(path, "r") as f:
        return json.load(f)

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Convert a NIfTI into a .vrdf for Unity DVR rendering.\n"
            "Modes:\n"
            " - labelmap : single discrete volume\n"
            " - continuous4d : continuous volumes (MRI/CT), split along T\n"
            " - labelmap_weighted4d : labels + tumor weights (fused or split)\n"
            " - multi_overlay4d : one overlay .vrdf per channel\n"
        )
    )

    p.add_argument("--nifti", required=True,
                   help="Path to input .nii/.nii.gz")

    p.add_argument("--mode", required=True,
                   choices=["labelmap","continuous4d","labelmap_weighted4d","multi_overlay4d"],
                   help="Export mode")

    p.add_argument("--vrdf-out", default="scene.vrdf",
                   help=("Output path/base. "
                         "continuous4d: generates base_tXX.vrdf; "
                         "labelmap_weighted4d: "
                         "  - default <base>_lw.vrdf (2 interleaved channels)\n"
                         "  - if --split-weight then <base>_labels.vrdf + <base>_weights.vrdf\n"
                         "multi_overlay4d: generates base_chX.vrdf"))

    p.add_argument("--config", default=None,
                   help=(
                        "Path to a user JSON.\n"
                        "Example:\n"
                        "{\n"
                        "  \"transfer_function\": {\n"
                        "    \"labels\": {\n"
                        "      \"0\": {\"name\":\"Background\",\"color\":[0,0,0],\"alpha\":0.0},\n"
                        "      \"1\": {\"name\":\"Brain\",\"color\":[0.8,0.8,0.8],\"alpha\":0.1},\n"
                        "      \"2\": {\"name\":\"Enhancing tumor\",\"color\":[0,1,0],\"alpha\":0.4},\n"
                        "      \"3\": {\"name\":\"Core tumor\",\"color\":[0,0,1],\"alpha\":0.5},\n"
                        "      \"4\": {\"name\":\"Glioma\",\"color\":[1,1,0],\"alpha\":1.0}\n"
                        "    }\n"
                        "  }\n"
                        "}\n"
                   ))

    p.add_argument("--split-weight", action="store_true",
                   help=("(labelmap_weighted4d only) "
                         "If set, export 2 separate files "
                         "<base>_labels.vrdf and <base>_weights.vrdf "
                         "instead of a single interleaved *_lw.vrdf."))

    p.add_argument("--debug-dump", action="store_true",
                   help="Also dump intermediate .raw/.json files")

    p.add_argument("--raw-out", default="volume.raw",
                   help="[debug only] base .raw path for simple modes")
    p.add_argument("--meta-out", default="volume_meta.json",
                   help="[debug only] base path for metadata JSON")
    p.add_argument("--tf-out",   default="transfer_function.json",
                   help="[debug only] base path for transfer function JSON")

    args = p.parse_args()
    return args

def main():
    args = parse_args()
    user_cfg = load_user_config(args.config)

    img = nib.load(args.nifti)
    vol_full = img.get_fdata(dtype=np.float32)
    affine = img.affine
    spacing_mm = compute_spacing_mm(affine)

    user_tf = None
    if user_cfg is not None and "transfer_function" in user_cfg:
        user_tf = user_cfg["transfer_function"]

    print(f"[INFO] Original volume shape: {vol_full.shape}")
    print(f"[INFO] MODE: {args.mode}")

    if args.mode == "labelmap":
        export_labelmap_case(
            vol_full=vol_full,
            affine=affine,
            spacing_mm=spacing_mm,
            user_tf=user_tf,
            vrdf_out=args.vrdf_out,
            debug_dump=args.debug_dump,
            raw_out=args.raw_out,
            meta_out=args.meta_out,
            tf_out=args.tf_out
        )

    elif args.mode == "continuous4d":
        export_continuous4d_case(
            vol_full=vol_full,
            affine=affine,
            spacing_mm=spacing_mm,
            user_tf=user_tf,
            vrdf_out=args.vrdf_out,
            debug_dump=args.debug_dump,
            raw_out=args.raw_out,
            meta_out=args.meta_out,
            tf_out=args.tf_out
        )

    elif args.mode == "labelmap_weighted4d":
        export_labelmap_weighted_case(
            vol_full=vol_full,
            affine=affine,
            spacing_mm=spacing_mm,
            user_tf=user_tf,
            vrdf_out=args.vrdf_out,
            debug_dump=args.debug_dump,
            split_weight=args.split_weight
        )

    elif args.mode == "multi_overlay4d":
        export_multi_overlay4d_case(
            vol_full=vol_full,
            affine=affine,
            spacing_mm=spacing_mm,
            user_tf=user_tf,
            vrdf_out=args.vrdf_out,
            debug_dump=args.debug_dump
        )

    else:
        raise ValueError("Unknown MODE.")

if __name__ == "__main__":
    main()
