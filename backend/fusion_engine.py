"""
Weighted Ensemble Meta Classifier
Logistic Regression stacking layer that fuses the three upstream model scores
(graph, behavioral, temporal) into a unified Fraud Confidence Score.

On startup: loads a saved model if present, otherwise trains on synthetic
PaySim-style calibration data and saves the model for reuse.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    confidence_score: float
    graph_score: float
    behavioral_score: float
    temporal_score: float
    graph_available: bool
    behavioral_available: bool
    temporal_available: bool
    modalities_used: int


def _generate_synthetic_training_data(n_samples: int = 2000) -> tuple:
    """
    Generate PaySim-style synthetic calibration data for the meta classifier.
    Simulates realistic correlations between the three modality scores and fraud labels.
    This mirrors what the PaySim validation split would produce after upstream models process it.
    """
    rng = np.random.default_rng(42)

    # --- Legitimate transactions (70% of dataset) ---
    n_legit = int(n_samples * 0.70)
    legit_graph = rng.beta(2, 8, n_legit)           # low graph score: sparse network
    legit_behavioral = rng.beta(2, 8, n_legit)       # low behavioral: consistent patterns
    legit_temporal = rng.beta(2, 8, n_legit)         # low temporal: normal velocity
    legit_labels = np.zeros(n_legit)

    # --- Fraudulent transactions (30% of dataset, intentionally oversampled for classifier) ---
    n_fraud = n_samples - n_legit

    # Smurfing / Structuring — high temporal, medium graph
    n_type1 = n_fraud // 4
    t1_g = rng.beta(4, 4, n_type1)
    t1_b = rng.beta(3, 5, n_type1)
    t1_t = rng.beta(7, 2, n_type1)

    # Layering / Mule Networks — high graph, high behavioral
    n_type2 = n_fraud // 4
    t2_g = rng.beta(8, 2, n_type2)
    t2_b = rng.beta(7, 2, n_type2)
    t2_t = rng.beta(5, 3, n_type2)

    # Account Takeover — high behavioral, low graph
    n_type3 = n_fraud // 4
    t3_g = rng.beta(2, 7, n_type3)
    t3_b = rng.beta(8, 2, n_type3)
    t3_t = rng.beta(6, 3, n_type3)

    # Velocity Fraud — very high temporal, moderate others
    n_type4 = n_fraud - n_type1 - n_type2 - n_type3
    t4_g = rng.beta(4, 5, n_type4)
    t4_b = rng.beta(4, 5, n_type4)
    t4_t = rng.beta(9, 1, n_type4)

    fraud_graph = np.concatenate([t1_g, t2_g, t3_g, t4_g])
    fraud_behavioral = np.concatenate([t1_b, t2_b, t3_b, t4_b])
    fraud_temporal = np.concatenate([t1_t, t2_t, t3_t, t4_t])
    fraud_labels = np.ones(n_fraud)

    X = np.column_stack([
        np.concatenate([legit_graph, fraud_graph]),
        np.concatenate([legit_behavioral, fraud_behavioral]),
        np.concatenate([legit_temporal, fraud_temporal]),
    ])
    y = np.concatenate([legit_labels, fraud_labels])

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


class MetaClassifier:
    def __init__(self, model_save_path: str):
        self.model_save_path = Path(model_save_path)
        self._pipeline: Pipeline | None = None

    def initialize(self):
        """Load saved model or train a new one from synthetic calibration data."""
        if self.model_save_path.exists():
            self._pipeline = joblib.load(self.model_save_path)
            logger.info(f"Loaded meta-classifier from {self.model_save_path}")
            return

        logger.info("No saved meta-classifier found. Training on synthetic calibration data...")
        self._train()

    def _train(self):
        X, y = _generate_synthetic_training_data()

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight="balanced",  # handle class imbalance
            )),
        ])
        self._pipeline.fit(X, y)

        # Cross-validation to log performance
        cv_scores = cross_val_score(self._pipeline, X, y, cv=5, scoring="f1")
        logger.info(
            f"Meta-classifier CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, self.model_save_path)
        logger.info(f"Meta-classifier saved to {self.model_save_path}")

    def retrain(self):
        """Force retrain — call this if upstream models are recalibrated."""
        if self.model_save_path.exists():
            self.model_save_path.unlink()
        self._train()

    def fuse(
        self,
        graph_score: float | None,
        behavioral_score: float | None,
        temporal_score: float | None,
    ) -> FusionResult:
        """
        Fuse sub-model scores into a single Fraud Confidence Score.
        Handles missing modalities (None = timed out upstream model).
        """
        if self._pipeline is None:
            raise RuntimeError("MetaClassifier not initialized. Call initialize() first.")

        graph_available = graph_score is not None
        behavioral_available = behavioral_score is not None
        temporal_available = temporal_score is not None

        # Fallback: impute missing scores with 0.5 (neutral, maximum uncertainty)
        g = graph_score if graph_available else 0.5
        b = behavioral_score if behavioral_available else 0.5
        t = temporal_score if temporal_available else 0.5

        modalities_used = sum([graph_available, behavioral_available, temporal_available])

        X = np.array([[g, b, t]])
        confidence = float(self._pipeline.predict_proba(X)[0][1])  # probability of fraud class

        # Penalize confidence if modalities are missing (uncertainty penalty)
        if modalities_used < 3:
            missing_penalty = 0.10 * (3 - modalities_used)
            confidence = max(0.0, confidence - missing_penalty)
            logger.warning(
                f"{3 - modalities_used} modality/modalities unavailable. "
                f"Applied uncertainty penalty. Adjusted confidence: {confidence:.4f}"
            )

        return FusionResult(
            confidence_score=round(confidence, 4),
            graph_score=g,
            behavioral_score=b,
            temporal_score=t,
            graph_available=graph_available,
            behavioral_available=behavioral_available,
            temporal_available=temporal_available,
            modalities_used=modalities_used,
        )
