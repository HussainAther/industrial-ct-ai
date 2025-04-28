# cv_utils/pre_processing.py
"""
OpenCV-based pre-processing functions for CT image enhancement.
"""
import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Enhance contrast using Adaptive Histogram Equalization (CLAHE).

    Args:
        image (np.ndarray): Grayscale input image
        clip_limit (float): Threshold for contrast limiting
        tile_grid_size (tuple): Size of grid for histogram equalization

    Returns:
        np.ndarray: Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply((image * 255).astype(np.uint8)) / 255.0

def denoise_image(image, method='gaussian', kernel_size=5):
    """
    Apply noise reduction to the image.

    Args:
        image (np.ndarray): Grayscale input image
        method (str): 'gaussian' | 'median' | 'bilateral'
        kernel_size (int): Kernel size for filtering

    Returns:
        np.ndarray: Denoised image
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur((image * 255).astype(np.uint8), kernel_size) / 255.0
    elif method == 'bilateral':
        return cv2.bilateralFilter((image * 255).astype(np.uint8), 9, 75, 75) / 255.0
    else:
        raise ValueError("Unknown denoising method.")

def detect_edges(image, low_thresh=50, high_thresh=150):
    """
    Detect edges using Canny edge detection.

    Args:
        image (np.ndarray): Grayscale input image
        low_thresh (int): Lower bound for hysteresis thresholding
        high_thresh (int): Upper bound for hysteresis thresholding

    Returns:
        np.ndarray: Binary edge map
    """
    edges = cv2.Canny((image * 255).astype(np.uint8), low_thresh, high_thresh)
    return edges / 255.0

def adaptive_threshold(image):
    """
    Apply adaptive thresholding to highlight structures.

    Args:
        image (np.ndarray): Grayscale input image

    Returns:
        np.ndarray: Binary thresholded image
    """
    thresh = cv2.adaptiveThreshold((image * 255).astype(np.uint8), 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh / 255.0

if __name__ == "__main__":
    print("Pre-processing module ready.")

