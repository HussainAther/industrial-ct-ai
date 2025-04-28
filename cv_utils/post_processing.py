# cv_utils/post_processing.py
"""
Post-processing functions for detecting and highlighting defects in CT images.
"""
import cv2
import numpy as np

def find_defect_contours(image, min_area=10):
    """
    Detect contours in a binary image, typically after thresholding or edge detection.

    Args:
        image (np.ndarray): Binary input image (edges or thresholded)
        min_area (int): Minimum area to consider a contour as valid defect

    Returns:
        List[np.ndarray]: List of contours
    """
    contours, _ = cv2.findContours((image * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return filtered

def draw_defect_contours(original_image, contours, color=(0, 0, 255), thickness=2):
    """
    Draw detected defect contours on the original image.

    Args:
        original_image (np.ndarray): Grayscale or BGR image
        contours (list): List of contours to draw
        color (tuple): BGR color for contours
        thickness (int): Line thickness

    Returns:
        np.ndarray: Image with contours drawn
    """
    if len(original_image.shape) == 2:
        output = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        output = (original_image * 255).astype(np.uint8).copy()
    
    cv2.drawContours(output, contours, -1, color, thickness)
    return output / 255.0

def bounding_boxes_from_contours(contours):
    """
    Generate bounding boxes from contours.

    Args:
        contours (list): List of contours

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes (x, y, w, h)
    """
    boxes = [cv2.boundingRect(c) for c in contours]
    return boxes

if __name__ == "__main__":
    print("Post-processing module ready.")

