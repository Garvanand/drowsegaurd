# DrowseGuard v2 — Adaptive Driver Drowsiness Detection

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![OpenCV 4.9](https://img.shields.io/badge/OpenCV-4.9-green)

## Problem Statement
Drowsy driving remains a serious road safety risk in India, and studies show fatigue-related inattention is a recurring factor in highway and long-haul crash outcomes highlighted in NCRB-aligned road safety discussions. DrowseGuard v2 focuses on early fatigue detection with adaptive, personalized computer vision signals so interventions can occur before microsleep events escalate.

## What's New In v2
- PERCLOS-based temporal drowsiness measurement
- Adaptive per-driver calibration for personalized EAR thresholding
- 3D head pose estimation for nodding and drooping detection
- PCA-compressed feature pipeline in the classifier
- Three-tier alert escalation with warning and critical levels
- Session-level CSV logging with critical-frame snapshots
- Analytics dashboard for historical session review
- Night-mode-aware preprocessing pipeline

## Architecture
```text
drowseguard/
├── main.py
├── app.py
├── preprocessor.py
├── face_detector.py
├── eye_analyzer.py
├── feature_extractor.py
├── classifier.py
├── alert_system.py
├── image_utils.py
├── config.py
├── perclos_engine.py
├── head_pose.py
├── fatigue_scorer.py
├── calibrator.py
├── session_logger.py
├── analytics_page.py
├── requirements.txt
└── README.md
```

## How It Works
1. Calibration: the system captures a 30-second baseline EAR profile and computes an adaptive threshold.
2. Detection Loop: each frame is preprocessed, landmarks are extracted, EAR and MAR are computed, PERCLOS is updated, and head pose is estimated.
3. Fatigue Scoring: a weighted composite score fuses PERCLOS, EAR deviation, blink behavior, yawn behavior, and head pitch.
4. Alerting: the alert engine escalates between normal, warning, and critical states with audio signaling and event logging.

## Course Module Coverage
| Module name | Specific functions |
| --- | --- |
| Module 1: Image Fundamentals and Enhancement | `preprocessor.preprocess_frame()`, `preprocessor.apply_clahe()`, `preprocessor.detect_night_mode()` |
| Module 2: Feature Extraction and Representation | `feature_extractor.extract_hog_features()`, `feature_extractor.extract_combined_features()` |
| Module 3: Segmentation, Edges, and Morphology | `preprocessor.apply_canny_edges()`, `preprocessor.segment_eye_binary()`, `preprocessor.apply_morphological_ops()` |
| Module 4: Statistical Inference and Temporal Metrics | `perclos_engine.PerclosEngine.compute()`, `fatigue_scorer.FatigueScorer.compute()`, `eye_analyzer.EyeStateTracker.get_blink_rate_per_min()` |
| Module 5: Detection and ML Integration | `face_detector.FaceDetector.detect()`, `head_pose.HeadPoseEstimator.estimate()`, `classifier.EyeStateClassifier.fit()` with `sklearn.decomposition.PCA` |

## Algorithms
PERCLOS is computed as the fraction of closed-eye frames in a rolling window:

$$
PERCLOS = \frac{\sum_{t=1}^{N} \mathbb{1}(EAR_t < 1.2 \cdot T_{EAR})}{N}
$$

EAR is computed from six eye landmarks:

$$
EAR = \frac{\|p_2 - p_6\|_2 + \|p_3 - p_5\|_2}{2 \cdot \|p_1 - p_4\|_2}
$$

Fatigue score is a weighted composite:

$$
S = 0.30S_{perclos} + 0.25S_{ear} + 0.15S_{blink} + 0.15S_{yawn} + 0.15S_{pose}
$$

## Installation
1. Clone the repository.
```bash
git clone https://github.com/your-username/drowseguard.git
```
2. Move into the project folder.
```bash
cd drowseguard
```
3. Install dependencies.
```bash
pip install -r requirements.txt
```
4. Run the application.
```bash
python main.py
```

## Usage
Use the Live detection page for real-time inference, calibration, and alerts. Use the Session analytics page to inspect historical metrics, events, and snapshots from previous runs.

## Session Data
Each run is stored under `sessions/<session_id>/` with `events.csv`, `metrics.csv`, and a `snapshots/` folder containing saved critical frames.

## Tech Stack
| Layer | Technology |
| --- | --- |
| Language | Python 3.10+ |
| Computer Vision | OpenCV, MediaPipe |
| Numerical Computing | NumPy, SciPy |
| Machine Learning | scikit-learn, PCA |
| UI and Analytics | Streamlit, Plotly, pandas |
| Alerts and Persistence | pygame, CSV logging, JPEG snapshots |

## Author
Garv Anand
