# VRDF-SDK  
**Volume Rendering Data Format — Cross-Platform SDK for Encoding, Decoding, and Visualization**

![License](https://img.shields.io/badge/license-Apache-blue.svg)
![Language](https://img.shields.io/badge/languages-Python%20%7C%20C%23-orange)
![Status](https://img.shields.io/badge/status-Active-brightgreen)
![Format](https://img.shields.io/badge/file%20format-.vrdf-lightgrey)

---

## Overview

**VRDF** (Volume Rendering Data Format) is a compact, open binary container for **3D volumetric datasets** used in medical, scientific, and industrial visualization.  
It bundles **voxels**, **metadata**, and **transfer functions** into a single portable file optimized for **real-time volume rendering**.

A `.vrdf` file contains:

- **Voxel data** (`float32`, X-fastest order)  
- **Metadata** (dimensions, spacing, affine matrix)  
- **Transfer function** (continuous or labelmap)  
- **Normalization info** (`p1`, `p99`, etc.)

The **VRDF-SDK** provides official tools for **encoding**, **decoding**, and **rendering** across platforms (Python ↔ Unity).

---

## Features

- Unified container for 3D volumetric datasets (`.vrdf`)
- Python encoder for NIfTI, MHD, RAW, etc.
- Unity runtime with GPU raymarching
- Transfer function customization via JSON config
- Supports **continuous**, **labelmap**, and **multi-channel** volumes
- Self-contained: voxels + metadata + LUT
- Cross-platform (Windows, Linux, macOS)

---

## Repository Structure

```
vrdf-sdk/
│
├── python/               # Encoder, decoder, CLI, examples
│   ├── encode.py        # Encode NIfTI -> .vrdf
│   ├── read_vrdf.py           # Inspect .vrdf files
│   ├── pyproject.toml
│   └── examples/
│       ├── config_custom_colors.json
│       └── brain_demo.nii.gz
│
├── unity/                # Unity runtime (C# + shaders)
│   ├── Scripts/
│   │   └── VolumeVRDFLoader.cs
│   ├── Shaders/
│   │   └── VolumeDVR.shader
│
└── README.md
```

---

## Getting Started

### 1️. Install dependencies

```bash
pip install nibabel numpy scikit-learn
```

---

### 2️. Encode a `.vrdf` file from a NIfTI

```bash
python encode.py   --nifti brats_00012_separated-t2f.nii   --mode multi_label_channels   --config config_custom_colors.json   --vrdf-out brain_scene.vrdf
```

This produces:
- `volume.raw` — raw voxel data  
- `volume_meta.json` — metadata  
- `transfer_function.json` — transfer function (colors, alpha)  
- `brain_scene.vrdf` — ✅ **single packaged file** (all-in-one)

---

### 3️. `.vrdf` File Structure

Each `.vrdf` file is binary but simple to parse.  
All integers are stored as **little-endian unsigned 64-bit** (`<Q`).

| Section | Description | Content |
|----------|--------------|----------|
| **Header** | Magic `VRDF0001` + total file size | 16 bytes |
| **Meta JSON block** | Metadata (`dim`, `spacing`, `affine`, etc.) | UTF-8 JSON |
| **Transfer Function block** | Color mapping (`type`, `entries`, etc.) | UTF-8 JSON |
| **Voxel block** | Volume data (`float32`) | `dimX * dimY * dimZ * 4` bytes |

Binary layout:

```
[8B magic][8B total_size]
[8B meta_len][meta_json...]
[8B tf_len][tf_json...]
[8B raw_len][raw_bytes...]
```

---

### 4️Example Transfer Function Config

#### `config_custom_colors.json`
```json
{
  "transfer_function": {
    "labels": {
      "0": {"name": "background", "color": [0.0, 0.0, 0.0], "alpha": 0.0},
      "1": {"name": "tissue_gray", "color": [0.8, 0.8, 0.8], "alpha": 0.1},
      "2": {"name": "tissue_green", "color": [0.0, 1.0, 0.0], "alpha": 0.4},
      "3": {"name": "tissue_blue", "color": [0.0, 0.0, 1.0], "alpha": 0.5},
      "4": {"name": "tissue_yellow", "color": [1.0, 1.0, 0.0], "alpha": 1.0}
    }
  }
}
```

You can use the same config for both `--mode labelmap` and `--mode multi_label_channels`.

---

### 5️Inspect a `.vrdf` file (Python)

Use the provided tool:

```bash
python read_vrdf.py brain_scene.vrdf
```

It prints:

```
[OK] Parsed brain_scene.vrdf
  Magic: VRDF0001
  Total size: 247.5 MB
  Volume: 240×240×155
  Mode: multi_label_channels
  Labels:
    0 → background
    1 → tissue_gray
    2 → tissue_green
    3 → tissue_blue
    4 → tissue_yellow
```

You can also visualize a slice if `matplotlib` is installed.

---

### 6️Load and Render in Unity

1. Place your `.vrdf` file in:
   ```
   Assets/StreamingAssets/
   ```
2. Add:
   - `VolumeVRDFLoader.cs`
   - `VolumeDVR.shader`
3. Assign the filename in the inspector.

Unity automatically:
- Parses metadata and TF JSON  
- Builds a `3D Texture` from voxels  
- Creates a `1D LUT` from TF  
- Performs **real-time raymarching**

---

## Example Use Cases

- MRI/CT visualization  
- Medical segmentation rendering (BraTS, AI masks)  
- Scientific simulation data  
- Educational & serious games  
- 🧑Research visualization pipelines  

---

## Roadmap

- [x] `.vrdf` single-file container  
- [x] Python encoder (NIfTI → VRDF)  
- [x] Labelmap / Continuous / Multi-channel support  
- [x] Unity runtime decoder  
- [ ] C++ reference parser  
- [ ] WebGL/WebGPU viewer  
- [ ] ZSTD/LZ4 compression for voxel blocks  
- [ ] Streaming support for large datasets  

---

## Contributing

Pull requests and issues are welcome!  
You can contribute by:
- Improving the Python or Unity SDKs  
- Adding new LUT presets  
- Extending the `.vrdf` format (compression, streaming, metadata fields)

---

## License

**Apache 2.0 License**  
© 2025 Guillaume Schneider and contributors.  
Use freely for research, education, or commercial visualization.
