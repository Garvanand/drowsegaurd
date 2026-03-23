import cv2
import numpy as np

import config
from image_utils import bgr_to_gray


def apply_gaussian_blur(gray_frame):
    return cv2.GaussianBlur(gray_frame, config.GAUSSIAN_KERNEL, 0)


def apply_clahe(gray_frame):
    clahe = cv2.createCLAHE(clipLimit=config.HIST_CLIP_LIMIT, tileGridSize=config.HIST_TILE_SIZE)
    return clahe.apply(gray_frame)


def apply_canny_edges(gray_frame):
    return cv2.Canny(gray_frame, config.CANNY_LOW, config.CANNY_HIGH)


def apply_morphological_ops(binary_frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def apply_histogram_equalization(gray_frame):
    return cv2.equalizeHist(gray_frame)


def preprocess_frame(frame):
    gray = bgr_to_gray(frame)
    blurred = apply_gaussian_blur(gray)
    enhanced = apply_clahe(blurred)
    return enhanced, frame


def extract_eye_region(gray_frame, landmarks, eye_indices, padding=5):
    if landmarks is None or len(landmarks) == 0:
        return np.array([]), (0, 0, 0, 0)
    coords = np.asarray(landmarks, dtype=np.int32)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    h, w = gray_frame.shape[:2]
    selected = []
    max_index = len(coords) - 1
    for idx in eye_indices:
        safe_idx = int(np.clip(idx, 0, max_index))
        selected.append(coords[safe_idx])
    selected = np.asarray(selected, dtype=np.int32)
    x_min = max(int(np.min(selected[:, 0])) - padding, 0)
    y_min = max(int(np.min(selected[:, 1])) - padding, 0)
    x_max = min(int(np.max(selected[:, 0])) + padding, w - 1)
    y_max = min(int(np.max(selected[:, 1])) + padding, h - 1)
    if x_max <= x_min or y_max <= y_min:
        return np.array([]), (x_min, y_min, 0, 0)
    roi = gray_frame[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max - x_min, y_max - y_min)


def segment_eye_binary(eye_roi):
    if eye_roi is None or eye_roi.size == 0:
        return np.array([])
    _, thresh = cv2.threshold(eye_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closed


def apply_hough_circles(gray_frame):
    circles = cv2.HoughCircles(
        gray_frame,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=18,
        minRadius=2,
        maxRadius=30,
    )
    if circles is None:
        return np.array([])
    return circles[0].astype(np.int32)
