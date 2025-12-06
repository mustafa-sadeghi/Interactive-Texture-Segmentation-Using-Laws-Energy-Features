# Interactive Texture Segmentation Using Laws’ Energy Features

An interactive texture segmentation pipeline based on **Laws’ Texture Energy Measures**.  
The system lets the user define texture classes via **ROI selection**, then segments the image by comparing Laws-based feature vectors to **ROI-derived prototypes** with per-class distance thresholds.

---

## Description

This project implements classical, feature-based texture segmentation for grayscale images using Laws’ masks.  
The pipeline builds 16 filters, reduces them to 9 canonical energy maps, standardizes features, learns prototypes from user-selected ROIs, and assigns each pixel/fragment to the nearest class under a δ-percentile acceptance threshold.  
Two operation modes are supported:

- **Pointwise (PW)** – per-pixel features, sharper boundaries  
- **Fragmental (FRAG)** – block-level features, more robust regions

No deep learning is used; the method is fully interpretable and classical.

---

## Features

- 16 Laws’ filters → 9 energy maps (feature reduction)
- Local illumination normalization by sliding-window mean subtraction
- Standardized 9D feature vectors per pixel or per fragment
- **Interactive ROI selection** for up to 9 texture classes
- Prototype learning using median features inside ROIs
- Per-class thresholds based on distance percentiles (e.g. 95th / 99th)
- Support for:
  - **Pointwise mode** (`MODE='pointwise'`)
  - **Fragmental mode** (`MODE='fragmental'`, block-based)
- High-contrast colormap for segmentation visualization
- Export of:
  - Energy feature grid
  - ROIs overlay
  - Final segmentation map
  - Summary of classified pixels

---

## Project Structure

```text
.
├── main.py              # Main implementation: LawsTextureSegmentation + interactive UI
├── Im501.jpg            # Example input image (others: Im502.jpg, Im503.jpg)
├── MV8-Sadeghi.pdf      # Full project report (algorithm + results)
└── (generated outputs)
    ├── Im501_PW_energy9_grid.png   # Example: energy maps (pointwise)
```
## Installation
```bash
pip install opencv-python numpy scipy scikit-learn matplotlib
```
## Interactive Controls

When the OpenCV window (**Interactive ROI Picker**) opens:

- **1–9** (also Persian / Arabic-Indic digits): select active class index (C0..C8)  
- **Left-click**: add a centered square ROI for the current class  
- **+ / =**: increase ROI size  
- **- / _**: decrease ROI size  
- **Backspace**: remove the last ROI for the current class  
- **R / r**: reset all ROIs  
- **Enter**: finalize ROIs and run segmentation  
- **Esc**: quit without segmentation  

The status bar at the top of the window shows the current class and ROI size.

---

## Modes: Pointwise vs Fragmental

### Pointwise mode (`MODE='pointwise'`)
- Each pixel receives a 9D feature vector computed from the standardized Laws energy maps.  
- Produces **crisper and more detailed boundaries**.

### Fragmental mode (`MODE='fragmental'`)
- The image is divided into blocks of size `FRAGMENT_SIZE × FRAGMENT_SIZE`.  
- Each block is represented by the **median feature vector** of its pixels.  
- Produces **smoother, more homogeneous regions** and is more robust to noise.

---

## Outputs

After pressing **Enter** and running the segmentation, the script saves:

- `*_energy9_grid.png` – 3×3 grid of the nine standardized Laws energy maps  
- `*_rois.png` – original image with color-coded ROIs  
- `*_segmentation.png` – final labeled segmentation map (unknown pixels shown as white)  

**Console log includes:**

- Percentage of pixels assigned to a class (known vs unknown)

---

## Documentation

For full mathematical details, algorithm explanation, thresholds, examples, and  
experimental results on images **Im501–Im503**, see:

**MV8-Sadeghi.pdf**


    ├── Im501_PW_rois.png           # Original image with ROIs overlay
    └── Im501_PW_segmentation.png   # Final segmentation result
