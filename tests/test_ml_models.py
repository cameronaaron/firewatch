"""
Tests for machine learning models and prediction services.

Tests:
- Fire spread prediction accuracy
- Risk scoring calculations
- Anomaly detection
- Weather impact analysis
- Integration tests for ML endpoints
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from app.ml.anomaly_detection import AnomalyDetector
from app.ml.fire_prediction import FireSpreadPredictor
from app.ml.risk_scoring import RiskScorer
from app.models.database import Fire, Location
from app.services.advanced_weather import AdvancedWeatherService


class TestFireSpreadPredictor:
    """Tests for fire spread prediction model."""

    def setup_method(self):
        """Initialize predictor for each test."""
        self.predictor = FireSpreadPredictor()

    def test_extract_fire_features(self):
        """Test feature extraction from fire data."""
        fire = Fire(
            fire_id="test_fire",
            name="Test Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=1000,
            percent_contained=50,
        )

        features = self.predictor.extract_fire_features(fire, [])

        assert "avg_spread_rate" in features
        assert "distance_range" in features
        assert "current_distance" in features
        assert "log_growth_acres_24h" in features
        assert all(isinstance(v, (int, float)) for v in features.values())

    def test_predict_spread_rate(self):
        """Test spread rate prediction."""
        fire = Fire(
            fire_id="test_fire",
            name="Test Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=1000,
            percent_contained=50,
        )

        spread_rate = self.predictor.predict_spread_rate(fire, [])

        assert isinstance(spread_rate, float)
        assert spread_rate >= 0.0

    def test_predict_direction_probabilities(self):
        """Test direction probability distribution."""
        fire = Fire(
            fire_id="test_fire",
            name="Test Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=1000,
            bearing_degrees=90,
        )

        probabilities = self.predictor.predict_direction_probabilities(fire, [])

        assert len(probabilities) == 8  # 8 compass directions
        assert all(0 <= v <= 1 for v in probabilities.values())
        # Should sum to approximately 1
        assert 0.95 < sum(probabilities.values()) <= 1.05

    def test_predict_impact_timeline(self):
        """Test impact timeline prediction."""
        fire = Fire(
            fire_id="test_fire",
            name="Test Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=1000,
        )
        history = [
            SimpleNamespace(
                timestamp=datetime.now(UTC) - timedelta(hours=2),
                distance_miles=1.0,
                bearing_degrees=45.0,
            ),
            SimpleNamespace(
                timestamp=datetime.now(UTC),
                distance_miles=4.0,
                bearing_degrees=90.0,
            ),
        ]

        timeline = self.predictor.predict_impact_timeline(fire, history, target_distance=2.0)

        assert timeline is not None
        assert "hours_until_impact" in timeline
        assert "arrival_time" in timeline
        assert "confidence" in timeline
        assert 0 <= timeline["confidence"] <= 1


class TestRiskScorer:
    """Tests for risk scoring model."""

    def setup_method(self):
        """Initialize risk scorer for each test."""
        self.scorer = RiskScorer()

    def test_calculate_fire_risk_score(self):
        """Test fire risk score calculation."""
        fire = Fire(
            fire_id="test_fire",
            name="Test Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=5000,
            percent_contained=50,
        )

        score = self.scorer.calculate_fire_risk_score(fire)

        assert 0 <= score <= 100

    def test_calculate_location_vulnerability(self):
        """Test location vulnerability scoring."""
        location = Location(
            location_id="test_loc",
            display_name="Test Location",
            latitude=37.0,
            longitude=-120.0,
        )

        fire = Fire(
            fire_id="test_fire",
            name="Test Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=5000,
            percent_contained=50,
        )

        vulnerability = self.scorer.calculate_location_vulnerability_score(
            location,
            fire,
            distance_miles=5.0,
        )

        assert 0 <= vulnerability <= 100

    def test_calculate_combined_threat_score(self):
        """Test combined threat score."""
        location = Location(
            location_id="test_loc",
            display_name="Test Location",
            latitude=37.0,
            longitude=-120.0,
        )

        fire = Fire(
            fire_id="test_fire",
            name="Test Fire",
            latitude=37.1,
            longitude=-119.9,
            acres_burned=5000,
            percent_contained=20,
        )

        threat = self.scorer.calculate_combined_threat_score(
            location,
            fire,
            distance_miles=5.0,
            spread_rate=2.0,
        )

        assert 0 <= threat <= 100

    def test_get_threat_description(self):
        """Test threat level descriptions."""
        descriptions = [
            (5, "Minimal"),
            (20, "Low"),
            (40, "Moderate"),
            (60, "High"),
            (80, "Very High"),
            (95, "Critical"),
        ]

        for score, expected in descriptions:
            assert self.scorer.get_threat_description(score) == expected

    def test_get_recommendations(self):
        """Test recommendation generation."""
        recommendations_low = self.scorer.get_recommendations(15)
        recommendations_high = self.scorer.get_recommendations(85)

        assert isinstance(recommendations_low, list)
        assert isinstance(recommendations_high, list)
        assert len(recommendations_low) > 0
        assert len(recommendations_high) > 0
        # High risk should have more urgent recommendations
        assert any("evacuate" in r.lower() for r in recommendations_high)


class TestAnomalyDetector:
    """Tests for anomaly detection model."""

    def setup_method(self):
        """Initialize anomaly detector for each test."""
        self.detector = AnomalyDetector(contamination=0.1)

    def test_extract_behavior_features(self):
        """Test feature extraction for anomaly detection."""
        fire = Fire(
            fire_id="test_fire",
            name="Test Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=1000,
            percent_contained=50,
        )

        features = self.detector.extract_fire_behavior_features(fire, [])

        assert "mean_spread_rate" in features
        assert "std_spread_rate" in features
        assert "acreage" in features
        assert all(isinstance(v, (int, float)) for v in features.values())

    def test_detect_anomaly_normal(self):
        """Test anomaly detection for normal fire."""
        fire = Fire(
            fire_id="test_fire",
            name="Test Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=1000,
            percent_contained=50,
        )

        result = self.detector.detect_anomaly(fire, [])

        assert isinstance(result, dict)
        assert "is_anomaly" in result
        assert "reasons" in result
        assert isinstance(result["is_anomaly"], bool)
        assert isinstance(result["reasons"], list)

    def test_check_obvious_anomalies(self):
        """Test detection of obvious anomalies."""
        # Create features with obvious anomaly
        features = {
            "mean_spread_rate": 100,  # Extreme
            "std_spread_rate": 0,
            "mean_direction_change": 0,
            "spread_acceleration": 0,
            "containment_pct": 50,
            "acreage": 1000,
            "current_distance": 10,
        }

        reasons = self.detector._check_obvious_anomalies(features)

        assert len(reasons) > 0
        assert any("spread rate" in r.lower() for r in reasons)

    def test_get_anomaly_insight(self):
        """Test anomaly insight generation."""
        fire = Fire(
            fire_id="test_fire",
            name="Test Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=100000,  # Large fire
            percent_contained=0,
        )
        self.detector.detect_anomaly = lambda fire, history: {
            "is_anomaly": True,
            "reasons": ["Extreme spread rate"],
        }

        insight = self.detector.get_anomaly_insight(fire, [])

        assert isinstance(insight, str)


class TestWeatherService:
    """Tests for the weather service's spread multiplier."""

    def test_calculate_fire_spread_multiplier(self):
        """Test fire spread multiplier calculation."""
        # Baseline conditions
        baseline = AdvancedWeatherService.calculate_fire_spread_multiplier(85, 40, 10)

        # Hotter conditions should increase multiplier
        hotter = AdvancedWeatherService.calculate_fire_spread_multiplier(95, 40, 10)
        assert hotter > baseline

        # Wetter conditions should decrease multiplier
        wetter = AdvancedWeatherService.calculate_fire_spread_multiplier(85, 70, 10)
        assert wetter < baseline

        # All multipliers should be in valid range
        assert 0.1 <= baseline <= 10.0
        assert 0.1 <= hotter <= 10.0
        assert 0.1 <= wetter <= 10.0


class TestMLIntegration:
    """Integration tests for ML services."""

    def test_ml_analysis_service_wires_its_models(self):
        from app.services.ml_analysis import MLFireAnalysisService

        service = MLFireAnalysisService()

        assert service.predictor is not None
        assert service.risk_scorer is not None
        assert service.anomaly_detector is not None
        assert service.weather_service is not None


class TestPredictionAccuracy:
    """Tests for prediction model accuracy."""

    def test_spread_rate_bounds(self):
        """Test that spread rate predictions are within reasonable bounds."""
        predictor = FireSpreadPredictor()

        fire = Fire(
            fire_id="test_fire",
            name="Test Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=10000,
            percent_contained=0,
        )

        spread_rate = predictor.predict_spread_rate(fire, [])

        # Reasonable bounds for wildfire spread
        # Most wildfires spread 0.5 - 5 miles per hour
        assert 0 <= spread_rate <= 20

    def test_threat_score_sensitivity(self):
        """Test that threat score is sensitive to fire changes."""
        scorer = RiskScorer()

        location = Location(
            location_id="test_loc",
            display_name="Test Location",
            latitude=37.0,
            longitude=-120.0,
        )

        # Small, contained fire
        small_fire = Fire(
            fire_id="small_fire",
            name="Small Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=100,
            percent_contained=80,
        )

        # Large, uncontained fire
        large_fire = Fire(
            fire_id="large_fire",
            name="Large Fire",
            latitude=37.0,
            longitude=-120.0,
            acres_burned=100000,
            percent_contained=10,
        )

        small_threat = scorer.calculate_combined_threat_score(
            location,
            small_fire,
            distance_miles=10,
        )

        large_threat = scorer.calculate_combined_threat_score(
            location,
            large_fire,
            distance_miles=10,
        )

        # Large fire should have higher threat
        assert large_threat > small_threat


def test_anomaly_scoring_failure_is_logged_not_swallowed(caplog):
    from types import SimpleNamespace

    from app.ml.anomaly_detection import AnomalyDetector

    class BrokenModel:
        def predict(self, rows):
            raise ValueError("feature width changed")

    detector = AnomalyDetector()
    detector.model = BrokenModel()
    detector.is_trained = True
    fire = SimpleNamespace(acres_burned=10.0, percent_contained=50.0, bearing_degrees=None)

    with caplog.at_level("WARNING", logger="app.ml.anomaly_detection"):
        result = detector.detect_anomaly(fire, [])

    assert result["is_anomaly"] is False
    assert "feature width changed" in caplog.text


def test_spread_model_failure_is_logged_and_falls_back(caplog):
    from types import SimpleNamespace

    class BrokenScaler:
        def transform(self, rows):
            raise ValueError("scaler expects 12 features")

    predictor = FireSpreadPredictor()
    predictor.is_trained = True
    predictor.scaler = BrokenScaler()
    fire = SimpleNamespace(
        acres_burned=100.0,
        percent_contained=0.0,
        latitude=34.0,
        longitude=-118.0,
        bearing_degrees=None,
        discovered_at=None,
        last_updated=None,
        extra_metadata={},
    )

    with caplog.at_level("WARNING", logger="app.ml.fire_prediction"):
        rate = predictor.predict_spread_rate(fire, [])

    assert rate >= 0.0
    assert "scaler expects 12 features" in caplog.text
