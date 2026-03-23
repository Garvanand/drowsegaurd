import cv2
import numpy as np


def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def bgr_to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def draw_landmarks(frame, points, color=(0, 255, 0), radius=1):
    if points is None:
        return frame
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), radius, color, -1)
    return frame


def draw_bounding_box(frame, x, y, w, h, color, thickness=2):
    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    return frame


def draw_status_text(frame, text, position, color, font_scale=0.7):
    cv2.putText(frame, str(text), (int(position[0]), int(position[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return frame


def draw_ear_bar(frame, ear_value, threshold, x=10, y=60):
    bar_width = 24
    bar_height = 120
    max_ear = max(threshold * 2.0, 1e-6)
    normalized = float(np.clip(ear_value / max_ear, 0.0, 1.0))
    fill_height = int(bar_height * normalized)
    top_left = (int(x), int(y))
    bottom_right = (int(x + bar_width), int(y + bar_height))
    cv2.rectangle(frame, top_left, bottom_right, (220, 220, 220), 2)
    fill_top = int(y + bar_height - fill_height)
    cv2.rectangle(frame, (int(x + 2), fill_top), (int(x + bar_width - 2), int(y + bar_height - 2)), (0, 255, 0), -1)
    threshold_pos = int(y + bar_height - (threshold / max_ear) * bar_height)
    cv2.line(frame, (int(x - 4), threshold_pos), (int(x + bar_width + 4), threshold_pos), (0, 0, 255), 2)
    return frame


def flip_frame(frame):
    return cv2.flip(frame, 1)


def overlay_alpha(frame, overlay, alpha=0.4):
    if overlay is None:
        return frame
    if frame.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
