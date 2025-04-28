# defect_detection/yolo_detector.py
"""
Wrapper for YOLOv8-based defect detection in industrial CT images.
"""
import cv2
import numpy as np
from ultralytics import YOLO

class YOLODefectDetector:
    def __init__(self, model_path, conf_threshold=0.3):
        """
        Initialize YOLOv8 detector.

        Args:
            model_path (str): Path to YOLOv8 .pt model file
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_defects(self, image):
        """
        Run defect detection on a CT image.

        Args:
            image (np.ndarray): Grayscale or BGR image (0-1 float or 0-255 uint8)

        Returns:
            List[dict]: Detected defects with bounding boxes and confidence
        """
        if image.max() <= 1.0:
            img_input = (image * 255).astype(np.uint8)
        else:
            img_input = image.copy()

        results = self.model(img_input, verbose=False, conf=self.conf_threshold)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': round(conf, 3)
            })
        return detections

    def draw_detections(self, image, detections, color=(0, 255, 0)):
        """
        Draw bounding boxes for detected defects.

        Args:
            image (np.ndarray): Original image
            detections (list): Output from detect_defects

        Returns:
            np.ndarray: Annotated image
        """
        output = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, f"Defect {det['confidence']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return output / 255.0

if __name__ == "__main__":
    print("YOLODefectDetector ready.")

