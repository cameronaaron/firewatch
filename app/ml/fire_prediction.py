"""Fire spread rate, direction, impact timeline and containment estimates.

Trained scikit-learn regressors are used when available; otherwise each
estimate falls back to an empirical formula over the fire's history.
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from app.ml.spread_model_registry import BASELINE_MODEL_NAME, build_spread_regressor

logger = logging.getLogger(__name__)


class FireSpreadPredictor:
    """Predicts fire spread behavior using ML models trained on historical patterns."""

    FEATURE_VECTOR_ORDER = (
        "avg_spread_rate",
        "distance_range",
        "avg_direction_change",
        "current_distance",
        "current_bearing",
        "time_hours",
        "num_updates",
        "recent_spread_rate",
        "spread_rate_trend",
        "spread_acceleration",
        "spread_volatility",
        "direction_change_volatility",
        "direction_drift",
        "movement_towards_ratio",
        "update_density",
        "sensor_speed_recent",
        "last_update_interval_hours",
        "containment_factor",
        "log_acres_burned",
        "distance_to_size_ratio",
        "log_growth_acres_24h",
    )

    def __init__(self) -> None:
        """Initialize the fire spread predictor."""
        self.spread_model_name = BASELINE_MODEL_NAME
        self.spread_model = build_spread_regressor(self.spread_model_name)
        self.direction_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_selection_summary: dict | None = None
        self.model_source = "heuristic_fallback"
        self.promoted_model_path: str | None = None

    def set_spread_model(self, model_name: str) -> None:
        """Swap in one supported spread-model candidate by name."""
        self.spread_model_name = model_name
        self.spread_model = build_spread_regressor(model_name)
        self.is_trained = False
        self.model_source = "benchmark_candidate"

    def extract_fire_features(
        self,
        fire: Any,
        history: Sequence[Any],
    ) -> dict:
        """Extract features from fire history for prediction."""
        if len(history) < 2:
            return self._default_features(fire)

        # Convert to polars DataFrame for analysis
        rows = [
            {
                "timestamp": getattr(h, "recorded_at", h.timestamp),
                "distance": h.distance_miles,
                "bearing": h.bearing_degrees,
            }
            for h in sorted(
                history,
                key=lambda item: getattr(item, "recorded_at", item.timestamp),
            )
        ]
        timestamps = [row["timestamp"] for row in rows]
        hours_elapsed = [
            (timestamp - timestamps[0]).total_seconds() / 3600 for timestamp in timestamps
        ]
        df = pl.DataFrame(rows).with_columns(pl.Series("hours_elapsed", hours_elapsed))

        distance_values = np.nan_to_num(np.asarray(df["distance"], dtype=float), nan=0.0)
        bearing_values = np.nan_to_num(np.asarray(df["bearing"], dtype=float), nan=0.0)
        elapsed_values = np.asarray(df["hours_elapsed"], dtype=float)
        delta_hours = np.diff(elapsed_values)

        valid_mask = delta_hours > 1e-9
        distance_deltas = np.diff(distance_values)
        if np.any(valid_mask):
            signed_rates = distance_deltas[valid_mask] / delta_hours[valid_mask]
        else:
            signed_rates = np.array([], dtype=float)
        abs_rates = np.abs(signed_rates)

        avg_spread_rate = float(np.mean(abs_rates)) if abs_rates.size else 0.0
        recent_slice = abs_rates[-3:] if abs_rates.size else np.array([], dtype=float)
        recent_spread_rate = float(np.mean(recent_slice)) if recent_slice.size else avg_spread_rate
        early_slice = abs_rates[:-3] if abs_rates.size > 3 else abs_rates
        early_spread_rate = float(np.mean(early_slice)) if early_slice.size else recent_spread_rate
        spread_rate_trend = float(recent_spread_rate - early_spread_rate)
        spread_acceleration = 0.0
        if abs_rates.size >= 2:
            latest_interval = float(delta_hours[valid_mask][-1]) if np.any(valid_mask) else 1.0
            if latest_interval <= 1e-9:
                latest_interval = 1.0
            spread_acceleration = float((abs_rates[-1] - abs_rates[-2]) / latest_interval)
        spread_volatility = float(np.std(abs_rates, ddof=0)) if abs_rates.size else 0.0

        wrapped_bearing_delta = np.diff(bearing_values)
        wrapped_bearing_delta = (wrapped_bearing_delta + 180.0) % 360.0 - 180.0
        abs_bearing_delta = np.abs(wrapped_bearing_delta)
        avg_direction_change = float(np.mean(abs_bearing_delta)) if abs_bearing_delta.size else 0.0
        direction_change_volatility = (
            float(np.std(abs_bearing_delta, ddof=0)) if abs_bearing_delta.size else 0.0
        )
        direction_drift = (
            float(np.mean(wrapped_bearing_delta)) if wrapped_bearing_delta.size else 0.0
        )

        movement_towards_ratio = (
            float(np.mean(distance_deltas < 0.0)) if distance_deltas.size else 0.0
        )
        time_hours = float(df["hours_elapsed"][-1])
        update_density = float(df.height / max(time_hours, 1.0))

        last_update_interval_hours = float(delta_hours[-1]) if delta_hours.size else 0.0

        speed_points = [float(getattr(item, "speed_mph", 0.0) or 0.0) for item in history[-3:]]
        sensor_speed_recent = float(np.mean(speed_points)) if speed_points else 0.0

        containment_pct = float(getattr(fire, "percent_contained", 0.0) or 0.0)
        containment_factor = max(0.0, min(1.0, (100.0 - containment_pct) / 100.0))
        acres_burned = float(getattr(fire, "acres_burned", 0.0) or 0.0)
        log_acres_burned = float(np.log1p(max(0.0, acres_burned)))
        distance_to_size_ratio = float(
            distance_values[-1] / (1.0 + np.sqrt(max(acres_burned, 0.0)))
        )
        growth_acres_24h = self._extract_growth_acres_24h(fire)
        log_growth_acres_24h = float(np.log1p(max(0.0, growth_acres_24h)))

        distance_range = float(np.max(distance_values) - np.min(distance_values))

        return {
            "avg_spread_rate": avg_spread_rate,
            "distance_range": distance_range,
            "avg_direction_change": avg_direction_change,
            "current_distance": float(distance_values[-1]),
            "current_bearing": float(bearing_values[-1]),
            "time_hours": time_hours,
            "num_updates": int(df.height),
            "recent_spread_rate": recent_spread_rate,
            "spread_rate_trend": spread_rate_trend,
            "spread_acceleration": spread_acceleration,
            "spread_volatility": spread_volatility,
            "direction_change_volatility": direction_change_volatility,
            "direction_drift": direction_drift,
            "movement_towards_ratio": movement_towards_ratio,
            "update_density": update_density,
            "sensor_speed_recent": sensor_speed_recent,
            "last_update_interval_hours": last_update_interval_hours,
            "containment_factor": containment_factor,
            "log_acres_burned": log_acres_burned,
            "distance_to_size_ratio": distance_to_size_ratio,
            "log_growth_acres_24h": log_growth_acres_24h,
        }

    def _default_features(self, fire: Any | None = None) -> dict:
        """Return default features when history is insufficient."""
        containment_pct = float(getattr(fire, "percent_contained", 0.0) or 0.0) if fire else 0.0
        containment_factor = max(0.0, min(1.0, (100.0 - containment_pct) / 100.0))
        acres_burned = float(getattr(fire, "acres_burned", 0.0) or 0.0) if fire else 0.0
        growth_acres_24h = self._extract_growth_acres_24h(fire) if fire else 0.0
        return {
            "avg_spread_rate": 0.0,
            "distance_range": 0.0,
            "avg_direction_change": 0.0,
            "current_distance": 0.0,
            "current_bearing": 0.0,
            "time_hours": 0.0,
            "num_updates": 0,
            "recent_spread_rate": 0.0,
            "spread_rate_trend": 0.0,
            "spread_acceleration": 0.0,
            "spread_volatility": 0.0,
            "direction_change_volatility": 0.0,
            "direction_drift": 0.0,
            "movement_towards_ratio": 0.0,
            "update_density": 0.0,
            "sensor_speed_recent": 0.0,
            "last_update_interval_hours": 0.0,
            "containment_factor": containment_factor,
            "log_acres_burned": float(np.log1p(max(0.0, acres_burned))),
            "distance_to_size_ratio": 0.0,
            "log_growth_acres_24h": float(np.log1p(max(0.0, growth_acres_24h))),
        }

    @staticmethod
    def _extract_growth_acres_24h(fire: Any | None) -> float:
        """Extract 24h acreage growth from direct attrs or persisted fire metadata."""
        if fire is None:
            return 0.0

        growth_value = getattr(fire, "growth_acres_24h", None)
        if growth_value is None:
            extra_metadata = getattr(fire, "extra_metadata", None)
            if isinstance(extra_metadata, dict):
                growth_value = extra_metadata.get("growth_acres_24h")

        try:
            return float(growth_value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def predict_spread_rate(
        self,
        fire: Any,
        history: Sequence[Any],
    ) -> float:
        """
        Predict fire spread rate in miles per hour.

        Uses Random Forest regression trained on fire behavior patterns.

        Returns:
            Predicted spread rate in miles/hour
        """
        features = self.extract_fire_features(fire, history)
        feature_vector = self._features_to_vector(features)

        if self.is_trained:
            try:
                feature_vector = self._coerce_feature_vector_for_model(feature_vector)
                prediction = self.spread_model.predict(self.scaler.transform([feature_vector]))[0]
                return max(0.0, float(prediction))
            except Exception as exc:
                logger.warning("Spread model failed, using the empirical estimate: %s", exc)

        # Fallback: empirical estimation based on containment
        if fire.percent_contained:
            base_rate = (100 - fire.percent_contained) / 50 * features["avg_spread_rate"]
        else:
            base_rate = features["avg_spread_rate"]

        return max(0.0, float(base_rate))

    def predict_direction_probabilities(
        self,
        fire: Any,
        history: Sequence[Any],
    ) -> dict[int, float]:
        """
        Predict probability distribution over 8 compass directions.

        Returns:
            Dictionary mapping direction (0-7, 0=N) to probability (0-1)
        """
        features = self.extract_fire_features(fire, history)
        current_bearing = features["current_bearing"] or getattr(fire, "bearing_degrees", 0) or 0

        # Create probability distribution centered on current direction
        directions = {}
        direction_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        for i, angle in enumerate(direction_angles):
            # Calculate angular distance
            angular_dist = abs(angle - current_bearing)
            angular_dist = min(angular_dist, 360 - angular_dist)

            # Gaussian distribution centered on current bearing
            probability = np.exp(-0.5 * (angular_dist / 45) ** 2)
            directions[i] = float(probability)

        # Normalize to sum to 1
        total = sum(directions.values())
        if total > 0:
            directions = {k: v / total for k, v in directions.items()}

        return directions

    def predict_impact_timeline(
        self,
        fire: Any,
        history: Sequence[Any],
        target_distance: float = 0.0,
    ) -> dict | None:
        """
        Predict when fire will reach specific distance from location.

        Args:
            fire: Fire object
            history: Historical distance data
            target_distance: Distance to predict arrival at (miles)
        """
        features = self.extract_fire_features(fire, history)
        spread_rate = self.predict_spread_rate(fire, history)

        if spread_rate <= 0:
            return None

        current_distance = features["current_distance"]
        distance_to_travel = current_distance - target_distance

        if distance_to_travel <= 0:
            return {
                "hours_until_impact": 0.0,
                "arrival_time": datetime.now().isoformat(),
                "confidence": 0.8,
            }

        hours_until_impact = distance_to_travel / spread_rate

        return {
            "hours_until_impact": float(hours_until_impact),
            "arrival_time": (datetime.now() + timedelta(hours=hours_until_impact)).isoformat(),
            "confidence": min(0.95, 0.5 + 0.05 * features["num_updates"]),
        }

    def predict_containment_time(
        self,
        fire: Any,
        history: Sequence[Any],
    ) -> dict | None:
        """Predict when fire will be fully contained based on current trends."""
        if not fire.percent_contained or fire.percent_contained >= 99:
            return {
                "hours_until_contained": 0.0,
                "estimated_contained_time": datetime.now().isoformat(),
                "confidence": 0.95,
            }

        # Estimate containment rate (percent per hour)
        features = self.extract_fire_features(fire, history)

        if features["time_hours"] == 0:
            return None

        containment_rate = fire.percent_contained / features["time_hours"]

        if containment_rate <= 0:
            return None

        percent_remaining = 100 - fire.percent_contained
        hours_until_contained = percent_remaining / containment_rate

        return {
            "hours_until_contained": float(hours_until_contained),
            "estimated_contained_time": (
                datetime.now() + timedelta(hours=hours_until_contained)
            ).isoformat(),
            "confidence": min(0.8, 0.3 + 0.1 * features["num_updates"]),
        }

    def _features_to_vector(self, features: dict) -> list[float]:
        """Convert features dictionary to numpy vector."""
        return [float(features.get(name, 0.0) or 0.0) for name in self.FEATURE_VECTOR_ORDER]

    def _coerce_feature_vector_for_model(self, vector: list[float]) -> list[float]:
        """Align vector length to the currently loaded model/scaler expectation."""
        expected_feature_count = getattr(self.scaler, "n_features_in_", None)
        if not isinstance(expected_feature_count, int) or expected_feature_count <= 0:
            return vector
        if len(vector) == expected_feature_count:
            return vector
        if len(vector) > expected_feature_count:
            return vector[:expected_feature_count]
        return [*vector, *([0.0] * (expected_feature_count - len(vector)))]

    def train(self, fire_samples: list[dict], *, model_name: str | None = None) -> None:
        """
        Train prediction models on fire behavior samples.

        Each sample should contain:
        - features: dict of extracted features
        - spread_rate: actual spread rate
        - direction: actual direction (0-7)
        """
        if not fire_samples:
            return

        if model_name:
            self.set_spread_model(model_name)

        X = []
        y_spread = []

        for sample in fire_samples:
            X.append(self._features_to_vector(sample.get("features", {})))
            y_spread.append(sample.get("spread_rate", 0.0))

        if X and y_spread:
            X_array = np.array(X)
            self.scaler.fit(X_array)
            X_scaled = self.scaler.transform(X_array)

            self.spread_model.fit(X_scaled, y_spread)
            self.is_trained = True
            self.model_source = "runtime_trained"

    def save_model_artifact(self, model_path: str, *, metadata: dict | None = None) -> None:
        """Persist the trained spread model and scaler to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save an untrained spread model")

        artifact_path = Path(model_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_name": self.spread_model_name,
            "spread_model": self.spread_model,
            "scaler": self.scaler,
            "saved_at": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
            "model_selection_summary": self.model_selection_summary,
        }
        joblib.dump(payload, artifact_path)
        self.promoted_model_path = str(artifact_path)
        self.model_source = "promoted_artifact"

    def load_model_artifact(self, model_path: str) -> bool:
        """Load a promoted spread model artifact if it exists."""
        artifact_path = Path(model_path)
        if not artifact_path.exists():
            return False

        payload = joblib.load(artifact_path)
        spread_model = payload.get("spread_model")
        scaler = payload.get("scaler")
        model_name = payload.get("model_name")
        if spread_model is None or scaler is None or not isinstance(model_name, str):
            raise ValueError(f"Invalid spread-model artifact at {artifact_path}")

        self.spread_model_name = model_name
        self.spread_model = spread_model
        self.scaler = scaler
        self.is_trained = True
        self.promoted_model_path = str(artifact_path)
        self.model_source = "promoted_artifact"
        if payload.get("model_selection_summary"):
            self.model_selection_summary = payload.get("model_selection_summary")
        logger.info("Loaded promoted spread model artifact from %s", artifact_path)
        return True

    def get_model_provenance(self) -> dict[str, object | None]:
        """Expose the active model lineage for API responses and debugging."""
        return {
            "model_name": self.spread_model_name,
            "is_trained": self.is_trained,
            "model_source": self.model_source,
            "promoted_model_path": self.promoted_model_path,
            "selection_summary": self.model_selection_summary,
        }
