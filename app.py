import collections
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import analytics_page
import config
from alert_system import AlertSystem
from calibrator import BaselineCalibrator
from classifier import EyeStateClassifier
from eye_analyzer import EyeStateTracker, compute_bilateral_ear, compute_mar
from face_detector import FaceDetector
from fatigue_scorer import FatigueScorer
from feature_extractor import extract_combined_features
from head_pose import HeadPoseEstimator
from image_utils import bgr_to_rgb, draw_bounding_box, draw_ear_bar, draw_landmarks, draw_status_text, overlay_alpha, resize_frame
from perclos_engine import PerclosEngine
from preprocessor import apply_canny_edges, extract_eye_region, preprocess_frame, segment_eye_binary
from session_logger import SessionLogger


st.set_page_config(page_title="DrowseGuard", layout="wide", page_icon="🛡")

if "page" not in st.session_state:
    st.session_state["page"] = "Live detection"
st.sidebar.radio("Navigation", ["Live detection", "Session analytics"], key="page")

if st.session_state["page"] == "Session analytics":
    st.title("DrowseGuard v2 Analytics")
    analytics_page.render_analytics()
    st.stop()

st.title("DrowseGuard v2")
st.markdown("Adaptive drowsiness detection with PERCLOS, head pose, fatigue scoring, and session analytics")

ear_threshold = st.sidebar.slider("Base EAR Threshold", min_value=0.1, max_value=0.5, value=float(config.EAR_THRESHOLD), step=0.01)
mar_threshold = st.sidebar.slider("MAR Threshold", min_value=0.3, max_value=0.9, value=float(config.MAR_THRESHOLD), step=0.01)
consec_frames = st.sidebar.slider("Consecutive Eye Frames", min_value=5, max_value=40, value=int(config.EAR_CONSEC_FRAMES), step=1)
show_edge_overlay = st.sidebar.toggle("Show Edge Detection Overlay", value=True)
show_landmarks = st.sidebar.toggle("Show Landmarks", value=True)
show_head_pose_axes = st.sidebar.toggle("Show Head Pose Axes", value=True)
enable_alert_sound = st.sidebar.toggle("Enable Alert Sound", value=True)

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
    st.session_state.ear_history = collections.deque(maxlen=300)
if "running" not in st.session_state:
    st.session_state.running = False
if "calibrator" not in st.session_state:
    st.session_state.calibrator = BaselineCalibrator(duration_seconds=config.CALIBRATION_DURATION)
if "perclos_engine" not in st.session_state:
    st.session_state.perclos_engine = PerclosEngine(window_seconds=config.PERCLOS_WINDOW_SECONDS, fps=30)
if "fatigue_scorer" not in st.session_state:
    st.session_state.fatigue_scorer = FatigueScorer()
if "head_pose_estimator" not in st.session_state:
    st.session_state.head_pose_estimator = HeadPoseEstimator(config.FRAME_WIDTH, config.FRAME_HEIGHT)
if "session_logger" not in st.session_state:
    st.session_state.session_logger = SessionLogger()
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "last_event_level" not in st.session_state:
    st.session_state.last_event_level = "normal"

summary = st.session_state.session_logger.get_summary()
with st.sidebar.expander("Session summary", expanded=False):
    st.write(summary)

cal_text = st.session_state.calibrator.status_text()
cal_color = "#16a34a" if st.session_state.calibrator.calibrated else "#f59e0b" if st.session_state.calibrator.is_running else "#64748b"
st.sidebar.markdown(
    f"<div style='padding:8px 10px;border-radius:8px;background:{cal_color};color:white;font-weight:600'>{cal_text}</div>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 2])

with col1:
    stframe = st.empty()

with col2:
    calibration_progress_ph = st.empty()
    calibration_status_ph = st.empty()
    ear_metric_ph = st.empty()
    mar_metric_ph = st.empty()
    blink_metric_ph = st.empty()
    perclos_metric_ph = st.empty()
    fatigue_metric_ph = st.empty()
    risk_badge_ph = st.empty()
    fatigue_gauge_ph = st.empty()
    ear_chart_ph = st.empty()

with st.expander("Frame Analysis", expanded=True):
    debug_col1, debug_col2, debug_col3 = st.columns(3)
    edge_debug_ph = debug_col1.empty()
    clahe_debug_ph = debug_col2.empty()
    binary_debug_ph = debug_col3.empty()

default_edge = np.zeros((120, 120, 3), dtype=np.uint8)
default_gray = np.zeros((120, 120), dtype=np.uint8)
default_binary = np.zeros((24, 24), dtype=np.uint8)
edge_debug_ph.image(bgr_to_rgb(default_edge), caption="Canny Edge Overlay", use_column_width=True)
clahe_debug_ph.image(default_gray, caption="CLAHE Enhanced Grayscale", use_column_width=True, clamp=True)
binary_debug_ph.image(default_binary, caption="Binary Otsu Left Eye", use_column_width=True, clamp=True)

btn_col1, btn_col2 = st.columns(2)
start_clicked = btn_col1.button("Start")
stop_clicked = btn_col2.button("Stop")

if start_clicked:
    st.session_state.running = True
if stop_clicked:
    st.session_state.running = False

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    failed_reads = 0
    runtime_status = st.empty()

    if not cap.isOpened():
        runtime_status.error("Webcam could not be opened. Close other apps using the camera and try again.")
        st.session_state.running = False
        cap.release()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            failed_reads += 1
            runtime_status.warning("Unable to read frame from webcam.")
            blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            stframe.image(bgr_to_rgb(blank), channels="RGB", use_column_width=True)
            edge_debug_ph.image(bgr_to_rgb(default_edge), caption="Canny Edge Overlay", use_column_width=True)
            clahe_debug_ph.image(default_gray, caption="CLAHE Enhanced Grayscale", use_column_width=True, clamp=True)
            binary_debug_ph.image(default_binary, caption="Binary Otsu Left Eye", use_column_width=True, clamp=True)
            if failed_reads >= 3:
                st.session_state.running = False
                break
            time.sleep(0.05)
            continue

        failed_reads = 0
        runtime_status.empty()
        st.session_state.frame_idx += 1
        frame = resize_frame(frame, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        processed_gray, is_night_mode = preprocess_frame(frame)

        detector = st.session_state.detector
        tracker = st.session_state.tracker
        classifier = st.session_state.classifier
        alert_system = st.session_state.alert_system
        calibrator = st.session_state.calibrator
        perclos_engine = st.session_state.perclos_engine
        fatigue_scorer = st.session_state.fatigue_scorer
        head_pose_estimator = st.session_state.head_pose_estimator
        session_logger = st.session_state.session_logger

        if not calibrator.is_running and not calibrator.calibrated:
            calibrator.start()

        landmarks_obj = detector.detect(frame)
        display_frame = frame.copy()

        ear = 0.0
        mar = 0.0
        perclos = 0.0
        fatigue_score = 0.0
        pitch = 0.0
        yaw = 0.0
        roll = 0.0
        blink_rate = tracker.get_blink_rate_per_min()
        yawn_rate = tracker.get_yawn_rate_per_min()
        face_overlay = np.zeros((120, 120, 3), dtype=np.uint8)
        left_eye_binary = np.zeros((24, 24), dtype=np.uint8)

        full_edges = apply_canny_edges(processed_gray)
        face_overlay = cv2.cvtColor(full_edges, cv2.COLOR_GRAY2BGR)
        h, w = processed_gray.shape[:2]
        cx1, cx2 = w // 2 - 24, w // 2 + 24
        cy1, cy2 = h // 2 - 12, h // 2 + 12
        cx1, cy1 = max(cx1, 0), max(cy1, 0)
        cx2, cy2 = min(cx2, w), min(cy2, h)
        center_eye = processed_gray[cy1:cy2, cx1:cx2]
        if center_eye.size > 0:
            left_eye_binary = segment_eye_binary(center_eye)

        if landmarks_obj is not None:
            all_indices = list(range(len(landmarks_obj.landmark)))
            all_coords = detector.get_landmark_coords(landmarks_obj, all_indices, frame.shape)
            left_eye_coords = detector.get_landmark_coords(landmarks_obj, config.LEFT_EYE_INDICES, frame.shape)
            right_eye_coords = detector.get_landmark_coords(landmarks_obj, config.RIGHT_EYE_INDICES, frame.shape)
            mouth_coords = detector.get_landmark_coords(landmarks_obj, config.MOUTH_INDICES, frame.shape)

            ear = compute_bilateral_ear(left_eye_coords, right_eye_coords)
            mar = compute_mar(mouth_coords)

            adaptive_threshold = calibrator.get_adaptive_threshold() if calibrator.calibrated else ear_threshold
            config.EAR_THRESHOLD = adaptive_threshold

            left_eye_roi, left_eye_bbox = extract_eye_region(processed_gray, all_coords, config.LEFT_EYE_INDICES)
            right_eye_roi, right_eye_bbox = extract_eye_region(processed_gray, all_coords, config.RIGHT_EYE_INDICES)
            if left_eye_roi.size > 0:
                left_eye_binary = segment_eye_binary(left_eye_roi)
                feature_source = left_eye_roi
            else:
                feature_source = processed_gray

            features = extract_combined_features(feature_source, ear, mar)
            classifier.predict_ensemble(features)
            tracker.update(ear, mar)
            blink_rate = tracker.get_blink_rate_per_min()
            yawn_rate = tracker.get_yawn_rate_per_min()

            perclos_engine.update(ear)
            perclos = perclos_engine.compute()

            try:
                pose = head_pose_estimator.estimate(landmarks_obj, frame.shape)
                pitch = pose["pitch"]
                yaw = pose["yaw"]
                roll = pose["roll"]
            except Exception:
                pitch, yaw, roll = 0.0, 0.0, 0.0

            baseline_ear = calibrator.baseline_ear
            fatigue_score = fatigue_scorer.compute(perclos, ear, baseline_ear, blink_rate, yawn_rate, pitch)

            if calibrator.progress() < 1.0 and not calibrator.calibrated:
                if enable_alert_sound:
                    alert_system.stop_alert()
            else:
                if enable_alert_sound:
                    alert_system.update(fatigue_score)
                else:
                    alert_system.stop_alert()

            session_logger.log_metrics(ear, mar, perclos, fatigue_score, pitch, roll, blink_rate)

            risk_level = fatigue_scorer.get_risk_level(fatigue_score)
            if risk_level == "critical":
                session_logger.log_event("critical", ear, perclos, fatigue_score, tracker.blink_count, int(yawn_rate))
                session_logger.save_snapshot(display_frame, "critical")
            elif risk_level == "warning":
                session_logger.log_event("warning", ear, perclos, fatigue_score, tracker.blink_count, int(yawn_rate))
            st.session_state.last_event_level = risk_level

            if show_landmarks:
                draw_landmarks(display_frame, all_coords, color=(0, 255, 0), radius=1)
            if show_head_pose_axes and show_landmarks:
                display_frame = head_pose_estimator.draw_pose_axes(display_frame, landmarks_obj, frame.shape)
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
                    display_frame[y : y + h, x : x + w] = face_overlay

                calibrator.update(ear)

        draw_status_text(display_frame, f"EAR: {ear:.3f}", (10, 25), (255, 255, 255), font_scale=0.65)
        draw_status_text(display_frame, f"MAR: {mar:.3f}", (10, 45), (255, 255, 255), font_scale=0.65)
        draw_status_text(display_frame, f"PITCH: {pitch:.1f}", (10, 65), (255, 255, 255), font_scale=0.65)
        draw_ear_bar(display_frame, ear, config.EAR_THRESHOLD, x=10, y=80)

        st.session_state.ear_history.append(float(ear))
        threshold_line = [config.EAR_THRESHOLD] * len(st.session_state.ear_history)
        ear_df = pd.DataFrame({"EAR": list(st.session_state.ear_history), "Threshold": threshold_line})

        calibration_progress_ph.progress(float(st.session_state.calibrator.progress()))
        calibration_status_ph.markdown(
            f"<div style='padding:10px;border-radius:8px;background:{'#16a34a' if st.session_state.calibrator.calibrated else '#f59e0b'};color:white;font-weight:600'>{st.session_state.calibrator.status_text()}</div>",
            unsafe_allow_html=True,
        )
        ear_metric_ph.metric("Current EAR", f"{ear:.3f}")
        mar_metric_ph.metric("Current MAR", f"{mar:.3f}")
        blink_metric_ph.metric("Blink Rate (/min)", f"{blink_rate:.0f}")
        perclos_metric_ph.metric("PERCLOS", f"{perclos * 100:.1f}%")
        fatigue_metric_ph.metric("Fatigue Score", f"{fatigue_score:.1f}", delta=f"{fatigue_scorer.get_score_trend():.2f}")

        risk_label = fatigue_scorer.get_risk_level(fatigue_score).upper()
        risk_color = "#16a34a" if risk_label == "NORMAL" else "#f59e0b" if risk_label == "WARNING" else "#dc2626"
        risk_badge_ph.markdown(
            f"<div style='padding:10px;border-radius:8px;background:{risk_color};color:white;text-align:center;font-weight:700'>{risk_label}</div>",
            unsafe_allow_html=True,
        )
        fatigue_gauge_ph.progress(float(np.clip(fatigue_score / 100.0, 0.0, 1.0)))
        ear_chart_ph.line_chart(ear_df)

        if left_eye_binary.size == 0:
            left_eye_binary = np.zeros((24, 24), dtype=np.uint8)
        if face_overlay.size == 0:
            face_overlay = np.zeros((120, 120, 3), dtype=np.uint8)

        edge_debug_ph.image(bgr_to_rgb(face_overlay), caption="Canny Edge Overlay", use_column_width=True)
        clahe_debug_ph.image(processed_gray, caption=f"CLAHE Enhanced Grayscale | Night Mode: {is_night_mode}", use_column_width=True, clamp=True)
        binary_debug_ph.image(left_eye_binary, caption="Binary Otsu Left Eye", use_column_width=True, clamp=True)

        stframe.image(bgr_to_rgb(display_frame), channels="RGB", use_column_width=True)
        time.sleep(0.01)

    cap.release()
    if not st.session_state.running:
        st.session_state.alert_system.stop_alert()
else:
    st.info("Press Start to begin live detection.")
