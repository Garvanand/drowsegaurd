# DrowseGuard

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
| Module name | Concepts applied |
| --- | --- |
| Module 1: Image Fundamentals and Preprocessing | Grayscale conversion, Gaussian smoothing, CLAHE enhancement in `preprocessor.preprocess_frame()` and `preprocessor.apply_clahe()` |
| Module 2: Feature Representation and Extraction | HOG descriptor and normalized feature vectors in `feature_extractor.extract_hog_features()` and `feature_extractor.extract_combined_features()` |
| Module 3: Edge and Morphological Processing | Canny edge detection and morphology in `preprocessor.apply_canny_edges()` and `preprocessor.apply_morphological_ops()` |
| Module 4: Classical Machine Learning for Vision | Synthetic data generation, scaling, KNN and Naive Bayes training in `classifier.EyeStateClassifier.fit()` |
| Module 5: Object Detection and Tracking Applications | Face landmark localization and ROI extraction via `face_detector.FaceDetector.detect()` and `face_detector.get_face_roi()` |

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
Use the Streamlit sidebar controls to tune EAR threshold, MAR threshold, and consecutive eye-closure frames for detection sensitivity. Toggle landmark rendering, edge overlay visualization, and sound alerts as needed. Press Start to begin webcam inference and Stop to terminate processing. Monitor live EAR and MAR metrics, blink count, drowsiness status, and the rolling EAR trend chart for behavior over time.

## Project Structure
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
├── requirements.txt
└── README.md
```

## Algorithm
Eye Aspect Ratio is computed from six eye landmarks as:

$$
EAR = \frac{\|p_2 - p_6\|_2 + \|p_3 - p_5\|_2}{2 \cdot \|p_1 - p_4\|_2}
$$

A lower EAR sustained over consecutive frames indicates prolonged eye closure and potential drowsiness.

## Tech Stack
| Layer | Technology |
| --- | --- |
| Language | Python 3.10+ |
| Computer Vision | OpenCV, MediaPipe |
| Numerical Computing | NumPy, SciPy |
| Machine Learning | scikit-learn |
| UI | Streamlit |
| Audio Alerts | pygame |
| Utilities | imutils, joblib |

## Author
Garv Anand
