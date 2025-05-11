# training/export_yolo_model.py

from ultralytics import YOLO
import os

# Path to the best.pt model (trained previously)
MODEL_PATH = "./training/runs/ct_defect_yolo/weights/best.pt"

# Output directory for exported models
EXPORT_DIR = "./training/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Load trained YOLOv8 model
print("📦 Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)

# Export to ONNX
print("🔄 Exporting to ONNX...")
model.export(format="onnx", dynamic=True, imgsz=256, simplify=True, device='cpu')
os.rename("best.onnx", os.path.join(EXPORT_DIR, "ct_defect.onnx"))

# Export to TorchScript
print("🔄 Exporting to TorchScript...")
model.export(format="torchscript", imgsz=256, device='cpu')
os.rename("best.torchscript", os.path.join(EXPORT_DIR, "ct_defect.torchscript"))

print("✅ Export complete!")
print(f"ONNX: {EXPORT_DIR}/ct_defect.onnx")
print(f"TorchScript: {EXPORT_DIR}/ct_defect.torchscript")

