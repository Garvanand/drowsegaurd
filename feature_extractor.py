import cv2
import numpy as np

try:
    from skimage.feature import hog as skimage_hog
except Exception:
    skimage_hog = None


def _ensure_gray(eye_roi):
    if eye_roi is None or eye_roi.size == 0:
        return np.zeros((24, 24), dtype=np.uint8)
    if len(eye_roi.shape) == 3:
        return cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    return eye_roi


def _l2_normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def extract_hog_features(eye_roi):
    gray = _ensure_gray(eye_roi)
    resized = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA)
    if skimage_hog is not None:
        features = skimage_hog(
            resized,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
            feature_vector=True,
        )
        flat = np.asarray(features, dtype=np.float32).flatten()
        return _l2_normalize(flat)
    hog = cv2.HOGDescriptor(
        _winSize=(24, 24),
        _blockSize=(8, 8),
        _blockStride=(4, 4),
        _cellSize=(4, 4),
        _nbins=9,
    )
    features = hog.compute(resized)
    if features is None:
        return np.zeros((900,), dtype=np.float32)
    flat = features.flatten().astype(np.float32)
    return _l2_normalize(flat)


def extract_pixel_features(eye_roi):
    gray = _ensure_gray(eye_roi)
    resized = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA)
    flat = resized.flatten().astype(np.float32)
    return flat / 255.0


def extract_combined_features(eye_roi, ear_value, mar_value):
    hog_features = extract_hog_features(eye_roi)
    scalar_features = np.asarray([ear_value, mar_value], dtype=np.float32)
    return np.concatenate([hog_features, scalar_features], axis=0)


def build_feature_matrix(eye_samples, ear_values, mar_values):
    rows = []
    for eye_roi, ear_value, mar_value in zip(eye_samples, ear_values, mar_values):
        rows.append(extract_combined_features(eye_roi, ear_value, mar_value))
    if len(rows) == 0:
        return np.empty((0, 902), dtype=np.float32)
    return np.vstack(rows).astype(np.float32)
