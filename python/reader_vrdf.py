#!/usr/bin/env python3
import sys
import struct
import json
import numpy as np

def read_uint64_le(f):
    """Lit un uint64 little-endian depuis le fichier f."""
    data = f.read(8)
    if len(data) != 8:
        raise IOError("Unexpected EOF while reading uint64.")
    return struct.unpack("<Q", data)[0]

def load_vrdf(path):
    """
    Charge un fichier .vrdf et renvoie:
    meta_dict, tf_dict, volume_np
    """
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"VRDF0001":
            raise ValueError(f"Bad magic header {magic!r}, expected b'VRDF0001'")
        total_size = read_uint64_le(f)

        meta_len = read_uint64_le(f)
        meta_bytes = f.read(meta_len)
        if len(meta_bytes) != meta_len:
            raise IOError("Unexpected EOF while reading meta block.")
        meta_dict = json.loads(meta_bytes.decode("utf-8"))

        tf_len = read_uint64_le(f)
        tf_bytes = f.read(tf_len)
        if len(tf_bytes) != tf_len:
            raise IOError("Unexpected EOF while reading TF block.")
        tf_dict = json.loads(tf_bytes.decode("utf-8"))

        raw_len = read_uint64_le(f)
        raw_bytes = f.read(raw_len)
        if len(raw_bytes) != raw_len:
            raise IOError("Unexpected EOF while reading RAW block.")

    dimX, dimY, dimZ = meta_dict["dim"]
    expected_raw_len = dimX * dimY * dimZ * 4  # float32
    if raw_len != expected_raw_len:
        raise ValueError(
            f"RAW length mismatch: header={raw_len}B vs expected {expected_raw_len}B "
            f"for volume {dimX}x{dimY}x{dimZ} float32"
        )

    volume_np = np.frombuffer(raw_bytes, dtype="<f4")  # little-endian float32
    volume_np = volume_np.reshape((dimX, dimY, dimZ), order="C")

    return {
        "magic": magic.decode("ascii", errors="replace"),
        "total_size": total_size,
        "meta": meta_dict,
        "tf": tf_dict,
        "volume": volume_np
    }

def summarize(vrdf_obj):
    """
    Affiche un résumé humainement lisible de ce qu'on a chargé.
    """
    meta = vrdf_obj["meta"]
    tf   = vrdf_obj["tf"]
    vol  = vrdf_obj["volume"]

    dimX, dimY, dimZ = meta["dim"]
    spacing = meta.get("spacing_mm", None)
    mode = meta.get("mode", "unknown")
    intensity_range = meta.get("intensity_range", None)

    print("[OK] Parsed VRDF")
    print(f"  Magic:        {vrdf_obj['magic']}")
    print(f"  Total size:   {bytes_to_human(vrdf_obj['total_size'])}")
    print(f"  Volume shape: {dimX}×{dimY}×{dimZ}  (np shape={vol.shape}, dtype={vol.dtype})")
    print(f"  Mode:         {mode}")
    print(f"  Spacing (mm): {spacing}")
    print(f"  Intensity:    {intensity_range}")
    print(f"  Endianness:   {meta.get('endianness','?')} (recorded)")
    print(f"  Order:        {meta.get('order','?')}")

    tf_type = tf.get("type", "unknown")
    print(f"  TF Type:      {tf_type}")
    origin = tf.get("origin", None)
    if origin:
        print(f"  TF Origin:    {origin}")

    if tf_type == "labelmap":
        print("  Labels:")
        for entry in tf.get("entries", []):
            lbl   = entry.get("label")
            name  = entry.get("name", None)
            color = entry.get("color")
            alpha = entry.get("alpha")
            if name is None:
                name = f"label_{lbl}"
            print(f"    {lbl} → {name}  color={color} alpha={alpha}")

    if tf_type == "continuous":
        curve = tf.get("curve", [])
        print(f"  Curve keypoints (sample):")
        for i, pt in enumerate(curve[0:5]):
            print(f"    x={pt['x']:.3f} color={pt['color']} alpha={pt['alpha']:.3f}")
        if len(curve) > 5:
            last = curve[-1]
            print(f"    ...")
            print(f"    x={last['x']:.3f} color={last['color']} alpha={last['alpha']:.3f}")

def bytes_to_human(n):
    """
    Transforme une taille en bytes en string lisible genre 247.5 MB.
    """
    units = ["B","KB","MB","GB","TB"]
    step = 1024.0
    x = float(n)
    for u in units:
        if x < step:
            return f"{x:.1f} {u}"
        x /= step
    return f"{x:.1f} PB"

def show_slice(volume_np, z_index):
    """
    Affiche une coupe Z en utilisant matplotlib (si dispo).
    Pas de LUT fancy ici, c'est juste debug.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib non disponible, impossible d'afficher la coupe.")
        return

    dimZ = volume_np.shape[2]
    if z_index < 0 or z_index >= dimZ:
        print(f"[WARN] z_index {z_index} hors limites [0,{dimZ-1}]")
        return

    slice_2d = volume_np[:, :, z_index].T
    plt.imshow(slice_2d, origin="lower")
    plt.title(f"Z slice {z_index}")
    plt.colorbar()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} file.vrdf [--show-slice Z]")
        sys.exit(1)

    vrdf_path = sys.argv[1]

    # parse optional flag --show-slice
    z_index_to_show = None
    if "--show-slice" in sys.argv:
        idx = sys.argv.index("--show-slice")
        if idx+1 >= len(sys.argv):
            print("Erreur: --show-slice attend un index Z")
            sys.exit(1)
        try:
            z_index_to_show = int(sys.argv[idx+1])
        except ValueError:
            print("Erreur: --show-slice doit être un entier.")
            sys.exit(1)

    vrdf_obj = load_vrdf(vrdf_path)
    summarize(vrdf_obj)

    if z_index_to_show is not None:
        show_slice(vrdf_obj["volume"], z_index_to_show)

if __name__ == "__main__":
    main()
