import collections
import time

import numpy as np

import config


class BaselineCalibrator:
    def __init__(self, duration_seconds=30):
        self.duration = duration_seconds
        self.ear_samples = []
        self.start_time = None
        self.calibrated = False
        self.baseline_ear = config.EAR_THRESHOLD + 0.05
        self.baseline_std = 0.02
        self.is_running = False
        self.recent_ears = collections.deque(maxlen=120)

    def start(self):
        self.start_time = time.time()
        self.is_running = True
        self.ear_samples = []
        self.calibrated = False

    def update(self, ear):
        if not self.is_running or self.calibrated:
            return
        if ear > 0.1:
            self.ear_samples.append(float(ear))
            self.recent_ears.append(float(ear))
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            self.finalize()

    def finalize(self):
        if len(self.ear_samples) < 50:
            self.baseline_ear = round(float(config.EAR_THRESHOLD + 0.05), 4)
            self.baseline_std = 0.02
            self.calibrated = True
            self.is_running = False
            return
        samples = sorted(self.ear_samples)
        trim = int(len(samples) * 0.1)
        trimmed = samples[trim:] if trim < len(samples) else samples
        self.baseline_ear = round(float(np.mean(trimmed)), 4)
        self.baseline_std = round(float(np.std(trimmed)), 4)
        self.calibrated = True
        self.is_running = False

    def get_adaptive_threshold(self):
        return max(self.baseline_ear - 2.5 * self.baseline_std, 0.18)

    def progress(self):
        if not self.is_running:
            return 1.0 if self.calibrated else 0.0
        return min((time.time() - self.start_time) / self.duration, 1.0)

    def status_text(self):
        if self.calibrated:
            return "Calibrated"
        if self.is_running:
            return f"Calibrating... {int(self.progress() * 100)}%"
        return "Not started"
