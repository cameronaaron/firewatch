"""
Anomaly detection for unusual fire behavior patterns.

Uses Isolation Forest algorithm to detect:
- Fires with unusual growth rates
- Anomalous containment patterns
- Unexpected direction changes
- Statistical outliers in behavior
"""

import logging
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from app.models.database import DistanceHistory, FireLike

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Flags unusual fire behavior with fixed thresholds and, once trained, an isolation forest."""

    def __init__(self, contamination: float = 0.1) -> None:
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (0.0-0.5)
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        self.is_trained = False
        self.mean = None
        self.std = None

    def extract_fire_behavior_features(
        self,
        fire: FireLike,
        history: list[DistanceHistory],
    ) -> dict:
        """
        Extract behavioral features for anomaly detection.

        Features:
        - Spread rate (miles/hour)
        - Direction changes per hour
        - Growth acceleration
        - Containment rate
        - Acreage per hour
        """
        if len(history) < 2:
            return self._default_features()

        rows = [
            {
                "timestamp": h.timestamp,
                "distance": h.distance_miles,
                "bearing": h.bearing_degrees,
            }
            for h in sorted(
                history,
                key=lambda item: item.timestamp,
            )
        ]
        timestamps = [h.timestamp for h in sorted(history, key=lambda item: item.timestamp)]
        hours_elapsed = np.array(
            [(timestamp - timestamps[0]).total_seconds() / 3600 for timestamp in timestamps],
            dtype=float,
        )
        df = pl.DataFrame(rows).with_columns(pl.Series("hours_elapsed", hours_elapsed))

        distances = np.array(df["distance"].to_list(), dtype=float)
        bearings = np.array(df["bearing"].to_list(), dtype=float)
        elapsed_deltas = np.diff(hours_elapsed)

        with np.errstate(divide="ignore", invalid="ignore"):
            spread_rates = np.diff(distances) / elapsed_deltas
            direction_changes = np.abs(np.diff(bearings)) / elapsed_deltas

        finite_spread_rates = spread_rates[np.isfinite(spread_rates)]
        finite_direction_changes = direction_changes[np.isfinite(direction_changes)]

        mean_spread = finite_spread_rates.mean() if finite_spread_rates.size else 0.0
        std_spread = finite_spread_rates.std() if finite_spread_rates.size else 0.0
        mean_direction = finite_direction_changes.mean() if finite_direction_changes.size else 0.0

        spread_accel = (
            np.diff(finite_spread_rates) if finite_spread_rates.size > 1 else np.array([])
        )
        mean_accel = spread_accel.mean() if spread_accel.size else 0.0

        return {
            "mean_spread_rate": float(mean_spread),
            "std_spread_rate": float(std_spread),
            "mean_direction_change": float(mean_direction),
            "spread_acceleration": float(mean_accel),
            "containment_pct": float(fire.percent_contained or 0),
            "acreage": float(fire.acres_burned),
            "current_distance": float(distances[-1]),
        }

    def _default_features(self) -> dict:
        """Return default features for insufficient history."""
        return {
            "mean_spread_rate": 0.0,
            "std_spread_rate": 0.0,
            "mean_direction_change": 0.0,
            "spread_acceleration": 0.0,
            "containment_pct": 0.0,
            "acreage": 0.0,
            "current_distance": 0.0,
        }

    def detect_anomaly(
        self,
        fire: FireLike,
        history: list[DistanceHistory],
    ) -> dict:
        """
        Detect if fire exhibits anomalous behavior.

        - is_anomaly: Boolean indicating if behavior is anomalous
        - anomaly_score: Score from -1 (anomaly) to +1 (normal)
        - reasons: List of reasons if anomalous
        """
        features = self.extract_fire_behavior_features(fire, history)
        feature_vector = self._features_to_vector(features)

        result: dict[str, Any] = {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "reasons": [],
            "features": features,
        }

        # Statistical thresholds for obvious anomalies
        obvious_anomalies = self._check_obvious_anomalies(features)
        if obvious_anomalies:
            result["is_anomaly"] = True
            result["reasons"].extend(obvious_anomalies)
            return result

        # ML-based anomaly detection if model is trained
        if self.is_trained:
            try:
                predictions = self.model.predict([feature_vector])
                anomaly_score = self.model.score_samples([feature_vector])[0]

                result["is_anomaly"] = predictions[0] == -1
                result["anomaly_score"] = float(anomaly_score)

                if result["is_anomaly"]:
                    result["reasons"].extend(self._explain_anomaly(features))
            except Exception as exc:
                logger.warning("Isolation forest scoring failed: %s", exc)

        return result

    def _check_obvious_anomalies(self, features: dict) -> list[str]:
        """Check for obvious statistical anomalies."""
        reasons = []

        # Extreme spread rate (> 20 miles/hour is unusual)
        if features["mean_spread_rate"] > 20:
            reasons.append(f"Extreme spread rate: {features['mean_spread_rate']:.1f} mi/hr")

        # Very high acceleration
        if abs(features["spread_acceleration"]) > 10:
            reasons.append(f"Unusual acceleration: {features['spread_acceleration']:.1f}")

        # Sudden direction changes (> 45 degrees per hour)
        if features["mean_direction_change"] > 45:
            reasons.append(f"Rapid direction changes: {features['mean_direction_change']:.1f}°/hr")

        # Impossible containment (> 100%)
        if features["containment_pct"] > 100:
            reasons.append("Containment exceeds 100%")

        return reasons

    def _explain_anomaly(self, features: dict) -> list[str]:
        """Provide human-readable explanation of anomaly."""
        reasons = []

        # Compare to expected ranges
        if features["mean_spread_rate"] > 10:
            reasons.append("Fire spreading faster than typical")
        elif features["mean_spread_rate"] < -1:
            reasons.append("Fire distance decreasing unusually fast")

        if features["spread_acceleration"] > 5:
            reasons.append("Acceleration increasing dramatically")
        elif features["spread_acceleration"] < -5:
            reasons.append("Spread rate slowing unexpectedly")

        if features["mean_direction_change"] > 30:
            reasons.append("Direction changing erratically")

        if features["acreage"] > 100000:
            reasons.append("Unusually large fire size")

        return reasons if reasons else ["Behavior pattern outside normal range"]

    def _features_to_vector(self, features: dict) -> list[float]:
        """Convert features dictionary to numpy array."""
        return [
            features["mean_spread_rate"],
            features["std_spread_rate"],
            features["mean_direction_change"],
            features["spread_acceleration"],
            features["containment_pct"],
            features["acreage"],
            features["current_distance"],
        ]

    def train(self, fire_samples: list[dict]) -> None:
        """
        Train anomaly detection model on fire samples.

        Each sample should contain:
        - features: dict of extracted features
        """
        if not fire_samples:
            return

        X = []
        for sample in fire_samples:
            X.append(self._features_to_vector(sample.get("features", {})))

        if X:
            X_array = np.array(X)
            self.model.fit(X_array)
            self.is_trained = True

    def get_anomaly_insight(
        self,
        fire: FireLike,
        history: list[DistanceHistory],
    ) -> str | None:
        """Get human-readable insight about fire behavior."""
        result = self.detect_anomaly(fire, history)

        if not result["is_anomaly"]:
            return None

        if not result["reasons"]:
            return "Fire behavior deviates from normal patterns - monitoring advised"

        return " | ".join(result["reasons"])
