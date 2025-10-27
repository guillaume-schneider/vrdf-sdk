import struct
import json
import numpy as np

MAGIC = b"VRDF0001"

class VRDFVolume:
    def __init__(self):
        self.meta = None           # dict
        self.tf = None             # dict
        self.data = None           # np.ndarray (float32)
        self.shape = None          # (X,Y,Z)
        self.mode = None           # labelmap / continuous / etc.
        self.spacing = None        # [sx,sy,sz]
        self.affine = None         # 4x4
        self.dtype = None
        self.intensity_range = None

    def summary(self):
        print(f"[VRDF] mode={self.mode} shape={self.shape} dtype={self.dtype}")
        if self.spacing is not None:
            print(f"       spacing_mm={self.spacing}")
        if self.intensity_range is not None:
            print(f"       range={self.intensity_range}")
        if self.tf is not None:
            tf_type = self.tf.get('type', 'unknown')
            print(f"       TF type={tf_type}, entries={len(self.tf.get('entries', []))}, curve={len(self.tf.get('curve', []))}")


def read_uint64_le(f):
    b = f.read(8)
    if len(b) != 8:
        raise EOFError("Unexpected EOF while reading uint64.")
    return struct.unpack("<Q", b)[0]


def read_vrdf(path):
    """Lit un fichier .vrdf ou .vrdfw et renvoie un VRDFVolume"""
    vol = VRDFVolume()

    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Bad magic {magic}, expected {MAGIC}")

        total_size = read_uint64_le(f)
        meta_len = read_uint64_le(f)
        meta_json = f.read(meta_len).decode("utf-8")

        tf_len = read_uint64_le(f)
        tf_json = f.read(tf_len).decode("utf-8")

        raw_len = read_uint64_le(f)
        raw_bytes = f.read(raw_len)

    meta = json.loads(meta_json)
    tf = json.loads(tf_json)

    dim = meta.get("dim", [1,1,1])
    sx, sy, sz = meta.get("spacing_mm", [1.0,1.0,1.0])
    dtype = meta.get("dtype", "float32")
    mode = meta.get("mode", "unknown")
    intensity_range = meta.get("intensity_range", [0,1])
    affine = np.array(meta.get("affine", np.eye(4))).reshape((4,4))

    dimX, dimY, dimZ = [int(x) for x in dim]
    expected_bytes = dimX * dimY * dimZ * 4
    if len(raw_bytes) != expected_bytes:
        raise ValueError(f"RAW size mismatch: got {len(raw_bytes)}, expected {expected_bytes}")

    data = np.frombuffer(raw_bytes, dtype=np.float32)
    data = data.reshape((dimX, dimY, dimZ), order="C")

    vol.meta = meta
    vol.tf = tf
    vol.data = data
    vol.shape = (dimX, dimY, dimZ)
    vol.mode = mode
    vol.spacing = [sx, sy, sz]
    vol.affine = affine
    vol.dtype = dtype
    vol.intensity_range = intensity_range

    return vol


def dump_tf_summary(tf):
    """Affiche un résumé simple de la TF"""
    tf_type = tf.get("type", "unknown")
    print(f"TF type: {tf_type}")

    if tf_type == "labelmap":
        entries = tf.get("entries", [])
        for e in entries:
            print(f"  label={e['label']:3.0f}  name={e.get('name','')}  "
                  f"color={e['color']}  alpha={e['alpha']:.2f}")
    elif tf_type == "continuous":
        curve = tf.get("curve", [])
        print(f"  {len(curve)} points in curve.")
        if curve:
            print("  First:", curve[0])
            print("  Mid:", curve[len(curve)//2])
            print("  Last:", curve[-1])
    else:
        print(json.dumps(tf, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vrdf_reader.py <path_to_vrdf>")
        sys.exit(1)

    path = sys.argv[1]
    vrdf = read_vrdf(path)
    vrdf.summary()
    dump_tf_summary(vrdf.tf)

    print(f"\n[DATA] voxel min={vrdf.data.min():.4f}, max={vrdf.data.max():.4f}, mean={vrdf.data.mean():.4f}")
