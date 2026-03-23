import collections

import numpy as np


class FatigueScorer:
    def __init__(self):
        self.score_history = collections.deque(maxlen=300)
        self.weights = dict(perclos=0.30, ear_deviation=0.25, blink_rate=0.15, yawn_rate=0.15, head_pose=0.15)

    def compute(self, perclos_value, current_ear, baseline_ear, blink_rate_per_min, yawn_rate_per_min, head_pose_pitch):
        perclos_score = min(perclos_value / 0.70, 1.0) * 100.0
        ear_deviation = max(0.0, baseline_ear - current_ear) / baseline_ear if baseline_ear > 0 else 0.0
        ear_score = min(ear_deviation / 0.4, 1.0) * 100.0
        normal_blink_rate = 15.0
        blink_score = min(abs(blink_rate_per_min - normal_blink_rate) / normal_blink_rate, 1.0) * 100.0
        yawn_score = min(yawn_rate_per_min / 3.0, 1.0) * 100.0
        pose_score = min(abs(head_pose_pitch) / 30.0, 1.0) * 100.0
        weighted = (
            self.weights["perclos"] * perclos_score
            + self.weights["ear_deviation"] * ear_score
            + self.weights["blink_rate"] * blink_score
            + self.weights["yawn_rate"] * yawn_score
            + self.weights["head_pose"] * pose_score
        )
        final = round(min(max(weighted, 0.0), 100.0), 1)
        self.score_history.append(final)
        return final

    def get_risk_level(self, score):
        if score >= 70:
            return "critical"
        if score >= 40:
            return "warning"
        return "normal"

    def get_score_trend(self):
        if len(self.score_history) < 60:
            return 0.0
        hist = list(self.score_history)
        return float(np.mean(hist[-30:]) - np.mean(hist[-60:-30]))

    def smoothed_score(self):
        if not self.score_history:
            return 0.0
        return round(float(np.mean(list(self.score_history)[-10:])), 1)
