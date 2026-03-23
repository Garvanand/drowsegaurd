# DrowseGuard v2 — Adaptive Driver Drowsiness Detection

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![OpenCV 4.9](https://img.shields.io/badge/OpenCV-4.9-green)

## Problem Statement
Driver drowsiness is a major road safety challenge in India, where long-distance travel, night driving, and fatigue contribute to preventable accidents on highways and urban roads. DrowseGuard addresses this issue with a real-time computer vision system that monitors eye closure and mouth opening patterns from a live webcam feed, detects drowsiness risk early, and provides immediate alerts to improve driver awareness and reduce crash probability.

## Features
- Real-time EAR and MAR computation using MediaPipe facial landmarks
- Dual classifier inference with KNN and Gaussian Naive Bayes ensemble
- Audio alert subsystem with file-based alarm and generated fallback beep
- Multi-stage preprocessing pipeline visualization for interpretability
- Interactive Streamlit dashboard with live metrics and controls

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
