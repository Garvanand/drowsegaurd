import numpy as np
from scipy.spatial import distance as dist

import config


def compute_ear(eye_coords):
    eye_coords = np.asarray(eye_coords, dtype=np.float32)
    if eye_coords.shape[0] < 6:
        return 1.0
    a = dist.euclidean(eye_coords[1], eye_coords[5])
    b = dist.euclidean(eye_coords[2], eye_coords[4])
    c = dist.euclidean(eye_coords[0], eye_coords[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


def compute_mar(mouth_coords):
    mouth_coords = np.asarray(mouth_coords, dtype=np.float32)
    if mouth_coords.shape[0] < 4:
        return 0.0
    a = dist.euclidean(mouth_coords[0], mouth_coords[1])
    c = dist.euclidean(mouth_coords[2], mouth_coords[3])
    if c == 0:
        return 0.0
    return a / c


def compute_bilateral_ear(left_coords, right_coords):
    return (compute_ear(left_coords) + compute_ear(right_coords)) / 2.0


def is_eye_closed(ear):
    return ear < config.EAR_THRESHOLD


def is_mouth_open(mar):
    return mar > config.MAR_THRESHOLD


class EyeStateTracker:
    def __init__(self):
        self.closed_frame_counter = 0
        self.blink_count = 0
        self.drowsy = False
        self.yawn_counter = 0
        self.yawning = False
        self.open_frame_counter = 0

    def update(self, ear, mar):
        if is_eye_closed(ear):
            self.closed_frame_counter += 1
            self.open_frame_counter = 0
            if self.closed_frame_counter >= config.EAR_CONSEC_FRAMES:
                self.drowsy = True
        else:
            if self.closed_frame_counter > 0:
                self.blink_count += 1
            self.closed_frame_counter = 0
            self.open_frame_counter += 1
            if self.open_frame_counter >= 10:
                self.drowsy = False

        if is_mouth_open(mar):
            self.yawn_counter += 1
            if self.yawn_counter >= config.MAR_CONSEC_FRAMES:
                self.yawning = True
        else:
            self.yawn_counter = 0
            self.yawning = False

        return self.drowsy
