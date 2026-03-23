import collections

import cv2
import numpy as np


class HeadPoseEstimator:
    MEDIAPIPE_NOSE_TIP = 1
    MEDIAPIPE_CHIN = 152
    MEDIAPIPE_LEFT_EYE_CORNER = 263
    MEDIAPIPE_RIGHT_EYE_CORNER = 33
    MEDIAPIPE_LEFT_MOUTH = 287
    MEDIAPIPE_RIGHT_MOUTH = 57

    def __init__(self, frame_width, frame_height):
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-165.0, 170.0, -135.0),
                (165.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float64,
        )
        focal_length = float(frame_width)
        self.camera_matrix = np.array(
            [
                [focal_length, 0.0, frame_width / 2.0],
                [0.0, focal_length, frame_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self.pitch_history = collections.deque(maxlen=30)
        self.roll_history = collections.deque(maxlen=30)

    def get_pose_landmarks(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        indices = [
            self.MEDIAPIPE_NOSE_TIP,
            self.MEDIAPIPE_CHIN,
            self.MEDIAPIPE_LEFT_EYE_CORNER,
            self.MEDIAPIPE_RIGHT_EYE_CORNER,
            self.MEDIAPIPE_LEFT_MOUTH,
            self.MEDIAPIPE_RIGHT_MOUTH,
        ]
        max_idx = len(landmarks.landmark) - 1
        points = []
        for idx in indices:
            safe_idx = int(np.clip(idx, 0, max_idx))
            lm = landmarks.landmark[safe_idx]
            x = float(np.clip(lm.x * w, 0, w - 1))
            y = float(np.clip(lm.y * h, 0, h - 1))
            points.append((x, y))
        return np.array(points, dtype=np.float32)

    def _solve(self, landmarks, frame_shape):
        image_points = self.get_pose_landmarks(landmarks, frame_shape).astype(np.float64)
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise ValueError("solvePnP failed")
        return image_points, rotation_vector, translation_vector

    def estimate(self, landmarks, frame_shape):
        image_points, rotation_vector, translation_vector = self._solve(landmarks, frame_shape)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch = np.degrees(np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2]))
        yaw = np.degrees(np.arctan2(-rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1] ** 2 + rotation_matrix[2][2] ** 2)))
        roll = np.degrees(np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0]))
        self.pitch_history.append(float(pitch))
        self.roll_history.append(float(roll))
        return {
            "pitch": round(float(pitch), 2),
            "yaw": round(float(yaw), 2),
            "roll": round(float(roll), 2),
            "rotation_vector": rotation_vector,
            "translation_vector": translation_vector,
            "image_points": image_points,
        }

    def is_nodding(self):
        if len(self.pitch_history) < 10:
            return False
        return abs(float(np.mean(self.pitch_history))) > 15.0

    def is_head_dropped(self):
        if len(self.pitch_history) < 20:
            return False
        return float(np.mean(list(self.pitch_history)[-20:])) > 20.0

    def draw_pose_axes(self, frame, landmarks, frame_shape):
        try:
            pose = self.estimate(landmarks, frame_shape)
            nose_tip = tuple(np.int32(pose["image_points"][0]))
            axis = np.float32([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
            projected, _ = cv2.projectPoints(axis, pose["rotation_vector"], pose["translation_vector"], self.camera_matrix, self.dist_coeffs)
            projected = projected.reshape(-1, 2).astype(np.int32)
            cv2.line(frame, nose_tip, tuple(projected[0]), (0, 0, 255), 2)
            cv2.line(frame, nose_tip, tuple(projected[1]), (0, 255, 0), 2)
            cv2.line(frame, nose_tip, tuple(projected[2]), (255, 0, 0), 2)
            return frame
        except Exception:
            return frame
