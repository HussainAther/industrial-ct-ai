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

DATA_DIR = "./synthetic_phantoms/demo_data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_DIR = os.path.join(DATA_DIR, "images")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# 3. Simulate acoustic property maps
def generate_synthetic_image():
    print("🧠 Generating synthetic image with fake defect...")
    img = np.zeros((256, 256), dtype=np.float32)
    cv2 = __import__('cv2')
    cv2.circle(img, (128, 128), 80, 0.4, -1)
    cv2.rectangle(img, (90, 140), (120, 170), 0.9, -1)  # Simulated defect
    plt.imsave(os.path.join(OUTPUT_DIR, "synthetic_defect.png"), img, cmap='gray')
    print("✅ Synthetic image saved.")

# Main execution
if __name__ == "__main__":
    download_phantom()
    decompress_raw_gz(os.path.join(RAW_DIR, "p_324402160.raw.gz"),
                      os.path.join(RAW_DIR, "p_324402160.raw"))
    generate_synthetic_image()
    print("🎉 Demo data generation complete.")

