import collections

import config


class PerclosEngine:
    def __init__(self, window_seconds=30, fps=30):
        self.window_size = window_seconds * fps
        self.frame_buffer = collections.deque(maxlen=self.window_size)
        self.perclos_history = collections.deque(maxlen=300)

    def update(self, ear):
        self.frame_buffer.append(1 if ear < config.EAR_THRESHOLD * 1.2 else 0)

    def compute(self):
        if len(self.frame_buffer) < 10:
            return 0.0
        perclos = sum(self.frame_buffer) / len(self.frame_buffer)
        self.perclos_history.append(perclos)
        return round(perclos, 4)

    def is_drowsy_by_perclos(self):
        return self.compute() > 0.35

    def get_trend(self):
        if len(self.perclos_history) < 60:
            return "insufficient data"
        history = list(self.perclos_history)
        mid = len(history) // 2
        first = sum(history[:mid]) / max(mid, 1)
        second = sum(history[mid:]) / max(len(history) - mid, 1)
        if second > first * 1.1:
            return "worsening"
        if abs(second - first) < 0.05:
            return "stable"
        return "improving"

    def reset(self):
        self.frame_buffer.clear()
        self.perclos_history.clear()
