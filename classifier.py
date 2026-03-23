import os

import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import config


class EyeStateClassifier:
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=config.KNN_NEIGHBORS)
        self.gnb = GaussianNB()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def generate_synthetic_training_data(self):
        rng = np.random.default_rng(42)
        n_open = 200
        n_closed = 200
        hog_dim = 900

        open_hog = rng.normal(0.08, 0.04, size=(n_open, hog_dim)).astype(np.float32)
        closed_hog = rng.normal(0.05, 0.04, size=(n_closed, hog_dim)).astype(np.float32)
        open_hog = np.clip(open_hog, -1.0, 1.0)
        closed_hog = np.clip(closed_hog, -1.0, 1.0)

        open_ear = rng.normal(0.30, 0.03, size=(n_open, 1)).astype(np.float32)
        closed_ear = rng.normal(0.18, 0.02, size=(n_closed, 1)).astype(np.float32)
        open_mar = rng.normal(0.42, 0.06, size=(n_open, 1)).astype(np.float32)
        closed_mar = rng.normal(0.44, 0.06, size=(n_closed, 1)).astype(np.float32)

        x_open = np.hstack([open_hog, open_ear, open_mar])
        x_closed = np.hstack([closed_hog, closed_ear, closed_mar])

        x = np.vstack([x_open, x_closed]).astype(np.float32)
        y = np.concatenate([np.zeros(n_open, dtype=np.int32), np.ones(n_closed, dtype=np.int32)])
        return x, y

    def fit(self):
        x, y = self.generate_synthetic_training_data()
        x_scaled = self.scaler.fit_transform(x)
        self.knn.fit(x_scaled, y)
        self.gnb.fit(x_scaled, y)
        self.is_fitted = True

    def _prepare(self, features):
        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if not self.is_fitted:
            self.fit()
        return self.scaler.transform(features)

    def predict_knn(self, features):
        x_scaled = self._prepare(features)
        return int(self.knn.predict(x_scaled)[0])

    def predict_gnb(self, features):
        x_scaled = self._prepare(features)
        return int(self.gnb.predict(x_scaled)[0])

    def predict_ensemble(self, features):
        x_scaled = self._prepare(features)
        p_knn = self.knn.predict_proba(x_scaled)
        p_gnb = self.gnb.predict_proba(x_scaled)
        p = (p_knn + p_gnb) / 2.0
        return int(np.argmax(p, axis=1)[0])

    def get_knn_confidence(self, features):
        x_scaled = self._prepare(features)
        probs = self.knn.predict_proba(x_scaled)[0]
        return float(np.max(probs))

    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.knn, os.path.join(path, "knn.pkl"))
        joblib.dump(self.gnb, os.path.join(path, "gnb.pkl"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        joblib.dump(self.is_fitted, os.path.join(path, "state.pkl"))

    def load_models(self, path):
        self.knn = joblib.load(os.path.join(path, "knn.pkl"))
        self.gnb = joblib.load(os.path.join(path, "gnb.pkl"))
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        self.is_fitted = bool(joblib.load(os.path.join(path, "state.pkl")))
