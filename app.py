from collections import deque
import time

import cv2
import numpy as np
import streamlit as st

import config
from alert_system import AlertSystem
from classifier import EyeStateClassifier
from eye_analyzer import EyeStateTracker, compute_bilateral_ear, compute_mar
from face_detector import FaceDetector
from feature_extractor import extract_combined_features
from image_utils import bgr_to_rgb, draw_bounding_box, draw_ear_bar, draw_landmarks, draw_status_text, overlay_alpha, resize_frame
from preprocessor import apply_canny_edges, extract_eye_region, preprocess_frame, segment_eye_binary


st.set_page_config(title="DrowseGuard", layout="wide", page_icon="🛡")
st.title("DrowseGuard")
st.markdown("Real-time driver drowsiness detection with classical computer vision and machine learning")

ear_threshold = st.sidebar.slider("EAR Threshold", min_value=0.1, max_value=0.5, value=0.25, step=0.01)
mar_threshold = st.sidebar.slider("MAR Threshold", min_value=0.3, max_value=0.9, value=0.6, step=0.01)
consec_frames = st.sidebar.slider("Consecutive Eye Frames", min_value=5, max_value=40, value=20, step=1)
show_edge_overlay = st.sidebar.toggle("Show Edge Detection Overlay", value=True)
show_landmarks = st.sidebar.toggle("Show Landmarks", value=True)
enable_alert_sound = st.sidebar.toggle("Enable Alert Sound", value=True)

config.EAR_THRESHOLD = ear_threshold
config.MAR_THRESHOLD = mar_threshold
config.EAR_CONSEC_FRAMES = consec_frames

if "tracker" not in st.session_state:
    st.session_state.tracker = EyeStateTracker()
if "detector" not in st.session_state:
    st.session_state.detector = FaceDetector()
if "alert_system" not in st.session_state:
    st.session_state.alert_system = AlertSystem()
if "classifier" not in st.session_state:
    st.session_state.classifier = EyeStateClassifier()
if "ear_history" not in st.session_state:
    st.session_state.ear_history = deque(maxlen=100)
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns([3, 2])

with col1:
    stframe = st.empty()

with col2:
    ear_metric_ph = st.empty()
    mar_metric_ph = st.empty()
    blink_metric_ph = st.empty()
    drowsy_metric_ph = st.empty()
    ear_progress_ph = st.empty()
    ear_chart_ph = st.empty()

with st.expander("Frame Analysis", expanded=True):
    debug_col1, debug_col2, debug_col3 = st.columns(3)
    edge_debug_ph = debug_col1.empty()
    clahe_debug_ph = debug_col2.empty()
    binary_debug_ph = debug_col3.empty()

start_clicked = st.button("Start")
stop_clicked = st.button("Stop")

if start_clicked:
    st.session_state.running = True
if stop_clicked:
    st.session_state.running = False

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    failed_reads = 0

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            failed_reads += 1
            if failed_reads >= 3:
                st.session_state.running = False
                break
            time.sleep(0.05)
            continue

        failed_reads = 0
        frame = resize_frame(frame, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        processed_gray, _ = preprocess_frame(frame)

        detector = st.session_state.detector
        tracker = st.session_state.tracker
        classifier = st.session_state.classifier
        alert_system = st.session_state.alert_system

        landmarks_obj = detector.detect(frame)

        ear = 0.0
        mar = 0.0
        is_drowsy = False
        face_overlay = np.zeros((120, 120, 3), dtype=np.uint8)
        left_eye_binary = np.zeros((24, 24), dtype=np.uint8)

        display_frame = frame.copy()

        if landmarks_obj is not None:
            all_indices = list(range(len(landmarks_obj.landmark)))
            all_coords = detector.get_landmark_coords(landmarks_obj, all_indices, frame.shape)
            left_eye_coords = detector.get_landmark_coords(landmarks_obj, config.LEFT_EYE_INDICES, frame.shape)
            right_eye_coords = detector.get_landmark_coords(landmarks_obj, config.RIGHT_EYE_INDICES, frame.shape)
            mouth_coords = detector.get_landmark_coords(landmarks_obj, config.MOUTH_INDICES, frame.shape)

            ear = compute_bilateral_ear(left_eye_coords, right_eye_coords)
            mar = compute_mar(mouth_coords)

            left_eye_roi, left_eye_bbox = extract_eye_region(processed_gray, all_coords, config.LEFT_EYE_INDICES)
            right_eye_roi, right_eye_bbox = extract_eye_region(processed_gray, all_coords, config.RIGHT_EYE_INDICES)

            if left_eye_roi.size > 0:
                left_eye_binary = segment_eye_binary(left_eye_roi)
                feature_source = left_eye_roi
            else:
                feature_source = processed_gray

            features = extract_combined_features(feature_source, ear, mar)
            predicted_state = classifier.predict_ensemble(features)
            tracked_state = tracker.update(ear, mar)
            is_drowsy = bool(tracked_state or predicted_state == 1)

            if enable_alert_sound:
                alert_system.update(is_drowsy)
            else:
                alert_system.stop_alert()

            if show_landmarks:
                draw_landmarks(display_frame, all_coords, color=(0, 255, 0), radius=1)

            if left_eye_bbox[2] > 0 and left_eye_bbox[3] > 0:
                draw_bounding_box(display_frame, left_eye_bbox[0], left_eye_bbox[1], left_eye_bbox[2], left_eye_bbox[3], (255, 255, 0), 1)
            if right_eye_bbox[2] > 0 and right_eye_bbox[3] > 0:
                draw_bounding_box(display_frame, right_eye_bbox[0], right_eye_bbox[1], right_eye_bbox[2], right_eye_bbox[3], (255, 255, 0), 1)

            face_roi, face_bbox = detector.get_face_roi(frame, landmarks_obj)
            if face_roi.size > 0:
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                edges = apply_canny_edges(face_gray)
                edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                face_overlay = overlay_alpha(face_roi, edge_bgr, alpha=0.45)
                if show_edge_overlay and face_bbox[2] > 0 and face_bbox[3] > 0:
                    x, y, w, h = face_bbox
                    display_frame[y:y + h, x:x + w] = face_overlay

        else:
            if enable_alert_sound:
                st.session_state.alert_system.update(False)

        status_text = "DROWSY" if is_drowsy else "ALERT"
        status_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        draw_status_text(display_frame, f"EAR: {ear:.3f}", (10, 25), (255, 255, 255), font_scale=0.65)
        draw_status_text(display_frame, f"MAR: {mar:.3f}", (10, 45), (255, 255, 255), font_scale=0.65)
        draw_status_text(display_frame, status_text, (10, 170), status_color, font_scale=0.8)
        draw_ear_bar(display_frame, ear, config.EAR_THRESHOLD, x=10, y=60)

        st.session_state.ear_history.append(ear)
        ear_series = np.array(st.session_state.ear_history, dtype=np.float32)

        ear_metric_ph.metric("Current EAR", f"{ear:.3f}")
        mar_metric_ph.metric("Current MAR", f"{mar:.3f}")
        blink_metric_ph.metric("Blink Count", f"{st.session_state.tracker.blink_count}")
        drowsy_metric_ph.metric(
            "Drowsy Status",
            "DROWSY" if is_drowsy else "ALERT",
            delta="High Risk" if is_drowsy else "Stable",
            delta_color="inverse" if is_drowsy else "normal",
        )
        ear_progress_ph.progress(float(np.clip(ear / 0.5, 0.0, 1.0)))
        ear_chart_ph.line_chart(ear_series)

        if left_eye_binary.size == 0:
            left_eye_binary = np.zeros((24, 24), dtype=np.uint8)
        if face_overlay.size == 0:
            face_overlay = np.zeros((120, 120, 3), dtype=np.uint8)

        edge_debug_ph.image(bgr_to_rgb(face_overlay), caption="Canny Edge Overlay", use_container_width=True)
        clahe_debug_ph.image(processed_gray, caption="CLAHE Enhanced Grayscale", use_container_width=True, clamp=True)
        binary_debug_ph.image(left_eye_binary, caption="Binary Otsu Left Eye", use_container_width=True, clamp=True)

        stframe.image(bgr_to_rgb(display_frame), channels="RGB", use_container_width=True)
        time.sleep(0.01)

    cap.release()
    if not st.session_state.running:
        st.session_state.alert_system.stop_alert()
else:
    st.info("Press Start to begin real-time detection.")
