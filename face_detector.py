import cv2
import mediapipe as mp
import numpy as np

import config


class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = None
        self.face_mesh = None
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.mp_face_mesh = mp.solutions.face_mesh
        else:
            try:
                from mediapipe.python.solutions import face_mesh

                self.mp_face_mesh = face_mesh
            except Exception:
                self.mp_face_mesh = None
        if self.mp_face_mesh is not None:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def detect(self, frame):
        if self.face_mesh is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def get_landmark_coords(self, landmarks, indices, frame_shape):
        h, w = frame_shape[:2]
        all_landmarks = landmarks.landmark
        max_idx = len(all_landmarks) - 1
        coords = []
        for idx in indices:
            safe_idx = int(np.clip(idx, 0, max_idx))
            lm = all_landmarks[safe_idx]
            x = int(np.clip(lm.x * w, 0, w - 1))
            y = int(np.clip(lm.y * h, 0, h - 1))
            coords.append([x, y])
        return np.asarray(coords, dtype=np.int32)

    def get_face_roi(self, frame, landmarks):
        h, w = frame.shape[:2]
        all_points = []
        for lm in landmarks.landmark:
            x = int(np.clip(lm.x * w, 0, w - 1))
            y = int(np.clip(lm.y * h, 0, h - 1))
            all_points.append([x, y])
        points = np.asarray(all_points, dtype=np.int32)
        x_min = int(np.min(points[:, 0]))
        y_min = int(np.min(points[:, 1]))
        x_max = int(np.max(points[:, 0]))
        y_max = int(np.max(points[:, 1]))
        pad_x = int((x_max - x_min) * 0.1)
        pad_y = int((y_max - y_min) * 0.1)
        x1 = max(x_min - pad_x, 0)
        y1 = max(y_min - pad_y, 0)
        x2 = min(x_max + pad_x, w - 1)
        y2 = min(y_max + pad_y, h - 1)
        if x2 <= x1 or y2 <= y1:
            return np.array([]), (0, 0, 0, 0)
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2 - x1, y2 - y1)

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()
