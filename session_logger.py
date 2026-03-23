import csv
import datetime
import os
import pathlib

import cv2


class SessionLogger:
    def __init__(self):
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enabled = True
        self.event_count = 0
        self.metrics_count = 0
        self.log_dir = pathlib.Path("sessions") / self.session_id
        self.snapshots_dir = self.log_dir / "snapshots"
        self.events_file = self.log_dir / "events.csv"
        self.metrics_file = self.log_dir / "metrics.csv"
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.snapshots_dir.mkdir(exist_ok=True)
            self._init_csv_files()
        except Exception:
            self.enabled = False

    def _init_csv_files(self):
        if not self.enabled:
            return
        with open(self.events_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event_type", "ear", "perclos", "fatigue_score", "blink_count", "yawn_count"])
        with open(self.metrics_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ear", "mar", "perclos", "fatigue_score", "pitch", "roll", "blink_rate"])

    def log_event(self, event_type, ear, perclos, fatigue_score, blink_count, yawn_count):
        if not self.enabled:
            return
        timestamp = datetime.datetime.now().isoformat()
        with open(self.events_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event_type, ear, perclos, fatigue_score, blink_count, yawn_count])
        self.event_count += 1

    def log_metrics(self, ear, mar, perclos, fatigue_score, pitch, roll, blink_rate):
        if not self.enabled:
            return
        if self.metrics_count % 30 == 0:
            timestamp = datetime.datetime.now().isoformat()
            with open(self.metrics_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, ear, mar, perclos, fatigue_score, pitch, roll, blink_rate])
        self.metrics_count += 1

    def save_snapshot(self, frame, label):
        if not self.enabled:
            return
        filename = self.snapshots_dir / f"{label}_{datetime.datetime.now().strftime('%H%M%S_%f')}.jpg"
        cv2.imwrite(str(filename), frame)

    def get_summary(self):
        snapshots_taken = 0
        if self.enabled and self.snapshots_dir.exists():
            snapshots_taken = len(list(self.snapshots_dir.glob("*.jpg")))
        return {
            "session_id": self.session_id,
            "total_events": self.event_count,
            "duration_seconds": int(self.metrics_count * 30 / 30),
            "snapshots_taken": snapshots_taken,
            "enabled": self.enabled,
        }
