# synthetic_phantoms/generate_demo_data.py
"""
Script to download, extract, convert, and generate synthetic CT demo data from the USCT Breast Phantom.
"""
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import gzip
import shutil
import requests
import json

DATA_DIR = "./synthetic_phantoms/demo_data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# 1. Download Phantom Files (Header + Raw Data)
def download_phantom():
    print("📦 Downloading sample phantom data...")
    urls = {
        "mhd": "https://dataverse.harvard.edu/api/access/datafile/4950808",
        "raw": "https://dataverse.harvard.edu/api/access/datafile/4950809"
    }
    files = {
        "mhd": os.path.join(RAW_DIR, "p_324402160.mhd"),
        "raw": os.path.join(RAW_DIR, "p_324402160.raw.gz")
    }
    for key in urls:
        if not os.path.exists(files[key]):
            r = requests.get(urls[key], stream=True)
            with open(files[key], 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    print("✅ Phantom files downloaded.")
    return files

# 2. Decompress .raw.gz

def decompress_raw_gz(gz_path, out_path):
    print("📂 Decompressing raw.gz file...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("✅ Decompression complete.")

# 3. Simulate acoustic property maps and annotations

def generate_synthetic_image():
    print("🧠 Generating synthetic image with fake defect and annotation...")
    img = np.zeros((256, 256), dtype=np.float32)
    cv2 = __import__('cv2')
    cv2.circle(img, (128, 128), 80, 0.4, -1)
    # Simulated defect
    top_left = (90, 140)
    bottom_right = (120, 170)
    cv2.rectangle(img, top_left, bottom_right, 0.9, -1)

    # Save image
    img_path = os.path.join(OUTPUT_DIR, "synthetic_defect.png")
    plt.imsave(img_path, img, cmap='gray')

    # Create annotation in YOLO-style format
    x_center = (top_left[0] + bottom_right[0]) / 2 / 256
    y_center = (top_left[1] + bottom_right[1]) / 2 / 256
    width = (bottom_right[0] - top_left[0]) / 256
    height = (bottom_right[1] - top_left[1]) / 256

    label_path = os.path.join(LABELS_DIR, "synthetic_defect.txt")
    with open(label_path, 'w') as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print("✅ Synthetic image and label saved.")

# Main execution
if __name__ == "__main__":
    download_phantom()
    decompress_raw_gz(os.path.join(RAW_DIR, "p_324402160.raw.gz"),
                      os.path.join(RAW_DIR, "p_324402160.raw"))
    generate_synthetic_image()
    print("🎉 Demo data generation complete.")

