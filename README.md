# VRDF-SDK  
**Volume Rendering Data Format — Cross-Platform SDK for Encoding, Decoding, and Visualization**

![License](https://img.shields.io/badge/license-Apache-blue.svg)
![Language](https://img.shields.io/badge/languages-Python%20%7C%20C%23-orange)
![Status](https://img.shields.io/badge/status-Active-brightgreen)
![Format](https://img.shields.io/badge/file%20format-.vrdf-lightgrey)

---

## Overview

**VRDF** (Volume Rendering Data Format) is a lightweight binary container designed for **3D volumetric datasets** used in scientific and medical visualization.  
It encapsulates everything needed for **direct volume rendering (DVR)** in a single, portable file.

A `.vrdf` file contains:

- **Voxel data** (`float32`, x-fastest order)
- **Metadata** (dimensions, spacing, affine transform)
- **Transfer function** (continuous or labelmap)
- **Normalization info** (`p1`, `p99`, etc.)

This repository — **`vrdf-sdk`** — provides the **official SDK** for reading, writing, and visualizing `.vrdf` files across platforms (Python ↔ Unity).

---

## Features

- Unified format for volumetric datasets (MRI, CT, segmentations, simulations)
- Python encoder (convert from NIfTI, MHD, RAW…)
- Unity/C# decoder for real-time GPU raymarching
- Supports both **continuous** and **labelmap** transfer functions
- Self-contained: voxels + metadata + LUT in one file
- Cross-platform (Linux, Windows, macOS)
- Extensible and open-spec

---

## Repository Structure

```
vrdf-sdk/
│
├── spec/                 # Format specification
│   └── vrdf1.0.md
│
├── python/               # Encoder and CLI tools
│   ├── encode_vrdf.py
│   ├── vrdf_writer.py
│   ├── requirements.txt
│   └── examples/
│
├── unity/                # Unity runtime (loader + DVR shader)
│   ├── Scripts/
│   │   └── VolumeVRDFLoader.cs
│   ├── Shaders/
│   │   └── VolumeDVR.shader
│   └── DemoScene.unity
│
├── examples/             # Demo datasets
│   ├── brain_tumor/
│   └── ct_abdomen/
│
└── README.md
```

---

## Getting Started

### Encode a `.vrdf` file (Python)

Install dependencies:

```bash
pip install nibabel numpy
```

Convert a NIfTI file into `.vrdf`:

```bash
python encode_vrdf.py   --input BraTS-GLI-00022-001-seg.nii.gz   --mode labelmap   --output volume.vrdf
```

Options:
- `--mode continuous` → raw MRI/CT data  
- `--mode labelmap` → segmentation mask  
- `--isotropic` → resample to isotropic voxels  

This produces a **single `.vrdf` file** that includes voxel data, metadata, and a color transfer function.

---

### Decode and visualize (Unity)

1. Copy your `.vrdf` file into:  
   `Assets/StreamingAssets/`
2. Add:
   - `VolumeVRDFLoader.cs` script
   - `VolumeDVR.shader` material
3. Assign your file name in the inspector.

Unity automatically:
- Parses the metadata and LUT from the embedded JSON  
- Builds a 3D Texture + 1D Transfer LUT  
- Renders the volume via GPU raymarching  

---

## `.vrdf` File Structure

| Section | Description | Size |
|----------|--------------|------|
| **Header** | Magic `VRDF1.0`, JSON length, voxel dtype | 40 bytes |
| **JSON Block** | Metadata + Transfer Function (UTF-8) | variable |
| **Voxel Block** | Raw float32 data (x-fastest) | `dimX * dimY * dimZ * 4` bytes |

Each `.vrdf` file is **self-contained**, platform-independent, and streamable.

---

## Transfer Functions

### Labelmap mode (for segmentation masks)
```json
"tf": {
  "type": "labelmap",
  "entries": [
    {"label":0, "color":[0,0,0], "alpha":0.0},
    {"label":1, "color":[0.9,0.3,0.3], "alpha":0.4},
    {"label":2, "color":[0.3,0.9,0.3], "alpha":0.3},
    {"label":4, "color":[1.0,0.8,0.0], "alpha":0.5}
  ]
}
```

### Continuous mode (for CT/MRI)
```json
"tf": {
  "type": "continuous",
  "curve": [
    {"x":0.0, "color":[0,0,0.3], "alpha":0.0},
    {"x":0.5, "color":[1,0.6,0.2], "alpha":0.3},
    {"x":1.0, "color":[1,1,1], "alpha":0.8}
  ]
}
```

---

## Example Use Cases

- **Medical imaging** (MRI/CT visualization)
- **Segmentation rendering** (BraTS, organ atlases, AI outputs)
- **Scientific data exploration**
- **Volumetric simulation results**
- **Educational or serious game content**

---

## Roadmap

- [ ] C++ reference parser  
- [ ] CLI tools (`vrdf inspect`, `vrdf convert`)  
- [ ] WebGL/WebGPU viewer  
- [ ] ZSTD/LZ4 compression for voxel blocks  
- [ ] Progressive streaming for large datasets  

---

## Contributing

Pull requests and issue reports are welcome!  
You can contribute by:
- Improving the Python or Unity SDKs
- Proposing new LUT presets
- Extending the `.vrdf` specification

---

## License

**Apache 2.0 License**  
© 2025 Guillaume Schneider and contributors.
