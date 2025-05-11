# training/train_yolo.py

import os
import random
import shutil
from ultralytics import YOLO

# Paths
DATA_ROOT = "./synthetic_phantoms/demo_data"
IMG_DIR = os.path.join(DATA_ROOT, "images")
LBL_DIR = os.path.join(DATA_ROOT, "labels")
YOLO_DATA = "./training/yolo_data"
TRAIN_DIR = os.path.join(YOLO_DATA, "images/train")
VAL_DIR = os.path.join(YOLO_DATA, "images/val")
TRAIN_LBL = os.path.join(YOLO_DATA, "labels/train")
VAL_LBL = os.path.join(YOLO_DATA, "labels/val")

# Ensure structure
for path in [TRAIN_DIR, VAL_DIR, TRAIN_LBL, VAL_LBL]:
    os.makedirs(path, exist_ok=True)

# 1. Split images and labels into train/val (80/20)
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])
random.shuffle(image_files)
split_idx = int(len(image_files) * 0.8)
train_imgs = image_files[:split_idx]
val_imgs = image_files[split_idx:]

def copy_split(files, src_img, src_lbl, dst_img, dst_lbl):
    for f in files:
        img_src = os.path.join(src_img, f)
        lbl_src = os.path.join(src_lbl, f.replace(".png", ".txt"))
        shutil.copy(img_src, os.path.join(dst_img, f))
        shutil.copy(lbl_src, os.path.join(dst_lbl, f.replace(".png", ".txt")))

copy_split(train_imgs, IMG_DIR, LBL_DIR, TRAIN_DIR, TRAIN_LBL)
copy_split(val_imgs, IMG_DIR, LBL_DIR, VAL_DIR, VAL_LBL)

# 2. Write YOLOv8 data.yaml
yaml_content = f"""
path: {YOLO_DATA}
train: images/train
val: images/val
names:
  0: defect
"""

with open(os.path.join(YOLO_DATA, "data.yaml"), 'w') as f:
    f.write(yaml_content.strip())

# 3. Train YOLOv8 model
print("🚀 Starting YOLOv8 training...")
model = YOLO("yolov8n.pt")  # Can change to yolov8s.pt or your own checkpoint
model.train(
    data=os.path.join(YOLO_DATA, "data.yaml"),
    epochs=30,
    imgsz=256,
    batch=8,
    name="ct_defect_yolo",
    project="./training/runs"
)

print("✅ Training complete! Model saved in ./training/runs/")

