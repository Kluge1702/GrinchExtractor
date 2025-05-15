# GrinchExtractor

A Python script for detecting and extracting the largest visually changed region ("Grinch") between two images — one with and one without the target object. This tool automatically generates cropped and transparent cutouts, as well as a version of the image with bounding boxes.

---

## Features

- Compares two images (with and without an object)
- Dynamically computes a threshold to detect changes
- Identifies the largest connected region (i.e., object)
- Outputs:
  - Cropped region of interest (`largest_grinch.png`)
  - Transparent object-only PNG (`bez_tla.png`)
  - Image with bounding box (`grinch_with_bbox.png`)

---

## Requirements

- Python 3.x  
- OpenCV  
- NumPy  

Install dependencies using pip:

```bash
pip install opencv-python numpy
