# synthetic_phantoms/augment_synthetic_dataset.py

import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# Output paths
DATA_DIR = "./synthetic_phantoms/demo_data"
IMG_DIR = os.path.join(DATA_DIR, "images")
LBL_DIR = os.path.join(DATA_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

# Parameters
IMG_SIZE = 256
NUM_SAMPLES = 20

def add_random_defect(img, defect_type="rectangle"):
    """
    Adds a defect of a specified type and returns:
    - modified image
    - bounding box in YOLO format
    """
    label = 0  # single-class
    if defect_type == "rectangle":
        w, h = random.randint(20, 40), random.randint(20, 40)
        x = random.randint(0, IMG_SIZE - w)
        y = random.randint(0, IMG_SIZE - h)
        cv2.rectangle(img, (x, y), (x + w, y + h), 0.9, -1)
    elif defect_type == "circle":
        r = random.randint(10, 25)
        x = random.randint(r, IMG_SIZE - r)
        y = random.randint(r, IMG_SIZE - r)
        cv2.circle(img, (x, y), r, 0.9, -1)
        w = h = 2 * r
    elif defect_type == "line":
        x1, y1 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        x2, y2 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        cv2.line(img, (x1, y1), (x2, y2), 0.9, 2)
        x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
    else:
        return img, None  # unknown defect

    # Normalize bbox to YOLO format
    xc = (x + w / 2) / IMG_SIZE
    yc = (y + h / 2) / IMG_SIZE
    wn = w / IMG_SIZE
    hn = h / IMG_SIZE
    return img, (label, xc, yc, wn, hn)

def generate_augmented_sample(idx):
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    cv2.circle(img, (IMG_SIZE//2, IMG_SIZE//2), 80, 0.4, -1)  # background

    num_defects = random.randint(1, 3)
    bboxes = []
    for _ in range(num_defects):
        defect_type = random.choice(["rectangle", "circle", "line"])
        img, bbox = add_random_defect(img, defect_type)
        if bbox:
            bboxes.append(bbox)

    # Apply optional noise/artifacts
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < 0.3:
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=10)

    # Save image
    img_path = os.path.join(IMG_DIR, f"synthetic_{idx:03d}.png")
    plt.imsave(img_path, img, cmap="gray")

    # Save label file
    label_path = os.path.join(LBL_DIR, f"synthetic_{idx:03d}.txt")
    with open(label_path, 'w') as f:
        for label, x, y, w, h in bboxes:
            f.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ Saved {img_path} with {len(bboxes)} defects.")

# Run the generator
if __name__ == "__main__":
    for i in range(NUM_SAMPLES):
        generate_augmented_sample(i)
    print("🎉 Augmented dataset generation complete.")

