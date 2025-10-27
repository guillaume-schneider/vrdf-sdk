# VRDF-SDK  
**Volume Rendering Data Format — Cross-Platform SDK for Encoding, Decoding, and Visualization**

![License](https://img.shields.io/badge/license-Apache-blue.svg)
![Language](https://img.shields.io/badge/languages-Python%20%7C%20C%23-orange)
![Status](https://img.shields.io/badge/status-Active-brightgreen)
![Format](https://img.shields.io/badge/file%20format-.vrdf-lightgrey)

---

## Overview

**VRDF (Volume Rendering Data Format)** is a compact binary container for 3D medical or scientific volumes, designed for **real-time volume rendering** (raymarching / DVR).

Each `.vrdf` file contains:

- **Voxel data** (`float32`)  
  - Memory order: `x-fast`, then `y`, then `z` (GPU scanline-friendly)
- **Metadata**
  - Dimensions (`dimX`, `dimY`, `dimZ`)
  - Voxel spacing in millimeters (`spacing_mm`)
  - 4×4 affine matrix (voxel→world, scanner alignment)
  - Intensity min / max range
  - Interpretation mode (continuous, labelmap, etc.)
- **Transfer function (TF)**  
  - Either a continuous LUT (intensity → color/alpha)
  - Or a labelmap LUT (label → color/alpha/human name)
- **Normalization info** (e.g. percentiles p1/p99)
- (Optional) channel semantics if multi-channel data

> Goal: one single self-contained file, readable in both **Python** and **Unity**, without separate `.raw`, `.json`, or `.lut` files.

---

## Why it’s useful

- Avoids the usual “NIfTI + mask + colors.txt + custom alpha script” nightmare.  
- Enables instant GPU raymarching visualization in Unity.  
- Encapsulates both continuous anatomy (MRI/CT) and discrete segmentations (AI masks, tumor maps).

---

## Core Features

### Self-contained container
- Voxels + metadata + transfer function stored in a single binary `.vrdf` file.

### Python Encoder
- `encode.py` converts NIfTI / 3D or 4D volumes → `.vrdf`
- Supports:
  - Single **fused file** (recommended)
  - Or multiple **legacy** specialized files

### Unity Runtime (C# + URP Shader)
- Loads `.vrdf` files directly
- Automatically reconstructs GPU `Texture3D` objects
- Builds LUTs (1D `Texture2D`) from the embedded TF
- Feeds data into the **raymarching shader** (`VolumeDVR.shader`)
- Handles transparency, label masking, and weight modulation at runtime

### Supported Data Types
- **continuous** — anatomical intensity (MRI/CT)
- **labelmap** — discrete segmentation (organs, tissues, lesions)
- **labelmap_weighted** — segmentation + confidence map per voxel
- **multi-overlay** — multiple overlay channels (edema, necrosis, etc.)
- **4D → split** — temporal or multi-channel volumes exported as multiple `.vrdf`

### Customization
- Simple JSON config for color, alpha, and name per label
- Optional Unity UI (via `VolumeDVR`) for per-class visibility, transparency, recoloring

### Cross-platform
- Python (Windows / Linux / macOS)
- Unity runtime (Windows / Linux / macOS builds)
- No proprietary dependencies

---

## Repository Structure

```text
vrdf-sdk/
│
├── python/
│   ├── encode.py              # CLI: NIfTI -> VRDF
│   ├── read_vrdf.py           # Inspect/debug a .vrdf file
│   ├── pyproject.toml
│   └── examples/
│       ├── config_custom_colors.json
│       └── brain_demo.nii.gz
│
├── unity/
│   ├── Scripts/
│   │   ├── VRDFLoader.cs      # Binary parser for .vrdf, builds GPU textures
│   │   └── VolumeDVR.cs       # Runtime component, shader setup, label UI
│   ├── Shaders/
│   │   └── VolumeDVR.shader   # Raymarching with weighted rendering
│
└── README.md
```

---

## Export Modes (Python Side)

The script `encode.py` accepts the `--mode` argument to define how to interpret and package a volume.

### 1. `labelmap`
- Input: discrete 3D volume (0 = background, 1 = tissue A, 2 = lesion B, …)
- Output:
  - `.vrdf` with `meta.mode = "labelmap"`
  - TF type `"labelmap"` = `(label → [r,g,b], alpha, name)` list
- Unity:
  - Per-class rendering (no interpolation)
  - Embedded per-label LUT

### 2. `continuous4d`
- Input: continuous 3D or 4D intensity volume (MRI, CT, PET, …)
  - If 4D → each timepoint exported separately: `..._t00.vrdf`, `..._t01.vrdf`, etc.
- Output:
  - Percentile normalization (p1/p99 → [0..1])
  - `meta.mode = "continuous"`
  - TF `"continuous"` = intensity → RGBA/alpha curve
- Unity:
  - Anatomical grayscale/soft shading rendering

### 3. `labelmap_weighted4d`
Input: 4D `(X,Y,Z,C)` volume, each channel `C` representing a region of interest (e.g. BraTS components: enhancement, edema, necrosis).

Automatically computes:
- Majority label per voxel (argmax across channels)
- Weight = normalized intensity (0–1)

Two export strategies:

#### a) Fused mode (default, recommended)
- Produces **one file**: `scene_lw.vrdf`
- For each voxel: two interleaved float32 values → `[label, weight01]`
- Metadata:
  - `meta.mode = "anatomy_label_weighted"`
  - `meta.channels = 2`
  - `meta.channel_meaning = ["labelmap","weight01"]`
- TF: `"labelmap"` type (color, alpha, name per class)
- Unity runtime:
  - Builds two `Texture3D` objects:
    - `_VolumeTexLabels` (point sampling)
    - `_VolumeTexWeights` (bilinear sampling)
  - `_VolumeTexWeights` modulates opacity → **weighted tumor rendering**

#### b) Split mode (`--split-weight`)
- Produces **two files**:
  - `scene_labels.vrdf` (`mode="anatomy_label"`)
  - `scene_weights.vrdf` (`mode="activity_weight"`)
- Unity still supports this legacy format.

### 4. `multi_overlay4d`
- Input: 4D `(X,Y,Z,C)` volume where each channel is an independent structure (Enhancing, Core, Edema, etc.)
- Output:
  - One `.vrdf` per channel (`scene_ch0.vrdf`, `scene_ch1.vrdf`, …)
  - Each with a simple `"continuous"` TF
- Unity:
  - Each overlay can be toggled independently

---

## Example Exports

### Simple labelmap with custom colors
```bash
python encode.py --nifti brain_segmentation.nii.gz --mode labelmap --config examples/config_custom_colors.json --vrdf-out brain_labels.vrdf
```

### Weighted tumor (single fused file)
```bash
python encode.py --nifti brats_case_42_multichannel.nii.gz --mode labelmap_weighted4d --vrdf-out scene_lw.vrdf
```

### Weighted tumor (split legacy version)
```bash
python encode.py --nifti brats_case_42_multichannel.nii.gz --mode labelmap_weighted4d --split-weight --vrdf-out scene.vrdf
# => scene_labels.vrdf + scene_weights.vrdf
```

---

## Binary Layout

All `.vrdf` files share a consistent little-endian layout:

```text
[8 bytes]    "VRDF0001"
[8 bytes]    total_size (uint64)

[8 bytes]    meta_len (uint64)
[meta_len]   meta_json (UTF-8)

[8 bytes]    tf_len (uint64)
[tf_len]     tf_json (UTF-8)

[8 bytes]    raw_len (uint64)
[raw_len]    raw_bytes (float32 data)
```

### Example `meta_json`

```json
{
  "dim": [240, 240, 155],
  "spacing_mm": [1.0, 1.0, 1.0],
  "dtype": "float32",
  "mode": "anatomy_label_weighted",
  "channels": 2,
  "channel_meaning": ["labelmap", "weight01"],
  "intensity_range": [0.0, 1.0],
  "affine": [
    [1.0,0.0,0.0,-120.0],
    [0.0,1.0,0.0,-120.0],
    [0.0,0.0,1.0,-75.0],
    [0.0,0.0,0.0,1.0]
  ],
  "order": "x-fast,y-then,z-outer",
  "endianness": "little"
}
```

### Example `tf_json`

```json
{
  "type": "labelmap",
  "entries": [
    {"label": 0, "name": "Background", "color": [0,0,0], "alpha": 0.0},
    {"label": 1, "name": "Enhancing Tumor", "color": [1,0,0], "alpha": 0.4},
    {"label": 2, "name": "Core Tumor", "color": [0,1,0], "alpha": 0.5},
    {"label": 3, "name": "Edema", "color": [0,0,1], "alpha": 0.4}
  ],
  "origin": "labelmap_weighted4d_default"
}
```

---

## Unity Pipeline

### 1. Place the files
Put `.vrdf` files into:
```text
Assets/StreamingAssets/
```
Examples:
- `scene_lw.vrdf` (fused label+weight)
- `scene_labels.vrdf` and `scene_weights.vrdf` (split mode)
- `t2_flair_t00.vrdf`, `t2_flair_t01.vrdf` (continuous4d)
- `scene_ch0.vrdf`, `scene_ch1.vrdf` (multi_overlay4d)

### 2. Scene setup
- Create a cube with a `MeshRenderer`
- Assign a `Material` using `VolumeDVR.shader`
- Add the `VolumeDVR.cs` component and configure:
  - `vrdfFusedFileName = "scene_lw.vrdf"` for fused mode
  - Or fill `vrdfLabelsFileName` / `vrdfWeightsFileName` for split mode

### 3. Runtime behavior
`VolumeDVR` uses `VRDFLoader.LoadFromFileSmart()` to parse `.vrdf` files:  
- Reads magic header `VRDF0001`
- Parses JSON blocks (`meta`, `tf`)
- Builds GPU textures:
  - Label texture (`Texture3D`, point filter)
  - Weight texture (`Texture3D`, bilinear filter)
  - 1D LUT (`Texture2D`) for colors/alpha
  - `_LabelCtrlTex` (256×1 RGBAFloat) for dynamic UI control

Injected shader uniforms:
```
_VolumeTexLabels
_VolumeTexWeights
_HasWeights
_TFTex
_LabelCtrlTex
```
Plus affine matrices and volume dimensions.

Result:  
- Real-time 3D raymarched volume rendering  
- Weight map drives transparency  
- Fully dynamic per-label color and visibility

### 4. Runtime Label Control API
```csharp
SetLabelVisible(int labelIndex, bool visible);
SetLabelOpacity(int labelIndex, float alpha01);
SetLabelTint(int labelIndex, Color tintRGB);
SoloLabel(int labelIndex);
ShowAll();
```

You can also query the TF labels dynamically:
```csharp
List<VolumeLabelInfoRuntime> infos = volumeDVR.GetLabelInfoList();
```

Each label entry provides:
- `labelIndex`
- `displayName`
- `color`
- `defaultVisible`

Perfect for dynamic medical or research visualization UIs.

---

## Python Inspection

To verify or debug a `.vrdf` export:

```bash
python read_vrdf.py scene_lw.vrdf
```

Output example:
```
[OK] Parsed scene_lw.vrdf
  Magic: VRDF0001
  Mode: anatomy_label_weighted
  Dim: 240x240x155
  Channels: 2 (labelmap, weight01)
  TransferFunction: labelmap (per-label RGBA)
  Labels present: [0,1,2,3,4]
```

---

## Roadmap

- [x] Self-contained `.vrdf` container (voxels + meta + TF)
- [x] Export modes: `labelmap`, `continuous4d`, `labelmap_weighted4d`, `multi_overlay4d`
- [x] Unity runtime supports fused `anatomy_label_weighted` mode
- [ ] C++ reference parser
- [ ] WebGL/WebGPU viewer
- [ ] ZSTD / LZ4 voxel block compression
- [ ] Streaming / bricking for >2GB datasets
- [ ] Unity UI widgets (label toggles, alpha sliders, clipping planes, etc.)

---

## License

**Apache 2.0 License**  
© 2025 Guillaume Schneider and contributors.  
Free for research, education, prototyping, and industrial visualization.
