import collections
import time

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
        self.ear_history = collections.deque(maxlen=60)
        self.blink_timestamps = collections.deque(maxlen=100)
        self.yawn_timestamps = collections.deque(maxlen=100)
        self.last_blink_ear = None
        self.last_blink_time = None
        self.blink_velocity = 0.0

    def update(self, ear, mar):
        now = time.time()
        self.ear_history.append(float(ear))

        if is_eye_closed(ear):
            self.closed_frame_counter += 1
            self.open_frame_counter = 0
            if self.closed_frame_counter >= config.EAR_CONSEC_FRAMES:
                self.drowsy = True
        else:
            if self.closed_frame_counter > 0:
                self.blink_count += 1
                self.blink_timestamps.append(now)
                if self.last_blink_ear is not None and self.last_blink_time is not None:
                    dt = max(now - self.last_blink_time, 1e-6)
                    self.blink_velocity = float((ear - self.last_blink_ear) / dt)
                self.last_blink_ear = float(ear)
                self.last_blink_time = now
            self.closed_frame_counter = 0
            self.open_frame_counter += 1
            if self.open_frame_counter >= 10:
                self.drowsy = False

        if is_mouth_open(mar):
            self.yawn_counter += 1
            if self.yawn_counter >= config.MAR_CONSEC_FRAMES and not self.yawning:
                self.yawning = True
                self.yawn_timestamps.append(now)
        else:
            self.yawn_counter = 0
            self.yawning = False

        return self.drowsy

    def get_blink_rate_per_min(self):
        now = time.time()
        cutoff = now - 60.0
        recent = [t for t in self.blink_timestamps if t >= cutoff]
        self.blink_timestamps = collections.deque(recent, maxlen=100)
        return float(len(recent))

    def get_yawn_rate_per_min(self):
        now = time.time()
        cutoff = now - 60.0
        recent = [t for t in self.yawn_timestamps if t >= cutoff]
        self.yawn_timestamps = collections.deque(recent, maxlen=100)
        return float(len(recent))

    def get_ear_trend(self):
        if len(self.ear_history) < 30:
            return 0.0
        values = list(self.ear_history)
        return float(np.mean(values[-15:]) - np.mean(values[-30:-15]))

    def get_ear_30s_mean(self):
        if not self.ear_history:
            return 0.0
        return float(np.mean(self.ear_history))
