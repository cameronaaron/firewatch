from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.ml.anomaly_detection import AnomalyDetector
from app.ml.benchmarking import (
    benchmark_algorithm_bakeoff,
    benchmark_spread_models,
    build_historical_spread_samples,
    build_production_readiness_scorecard,
    evaluate_model_promotion_gate,
)
from app.ml.fire_prediction import FireSpreadPredictor
from app.ml.risk_scoring import RiskScorer
from app.models.database import DistanceHistory, Fire, Location
from app.services.advanced_weather import AdvancedWeatherService, WeatherData
from app.services.comprehensive_data import ComprehensiveDataService
from app.services.fire_data import FireDataService


def make_fire(*, acres=500.0, contained=25.0, bearing=0.0) -> Fire:
    fire = Fire(
        fire_id="math-fire",
        name="Math Fire",
        latitude=34.0,
        longitude=-118.0,
        acres_burned=acres,
        percent_contained=contained,
    )
    fire.bearing_degrees = bearing
    return fire


def make_location(*, location_id="math-loc", latitude=35.0, longitude=-118.0) -> Location:
    return Location(
        location_id=location_id,
        address="123 Test St",
        display_name="Math Location",
        icon="🏠",
        latitude=latitude,
        longitude=longitude,
    )


def build_history():
    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    return [
        SimpleNamespace(timestamp=base_time, distance_miles=1.0, bearing_degrees=0.0),
        SimpleNamespace(
            timestamp=base_time + timedelta(hours=1), distance_miles=4.0, bearing_degrees=90.0
        ),
        SimpleNamespace(
            timestamp=base_time + timedelta(hours=3), distance_miles=10.0, bearing_degrees=180.0
        ),
    ]


def test_fire_spread_feature_math_exact_values():
    predictor = FireSpreadPredictor()
    fire = make_fire()
    fire.extra_metadata = {"growth_acres_24h": 144.0}
    features = predictor.extract_fire_features(fire, build_history())

    assert features["avg_spread_rate"] == pytest.approx(3.0)
    assert features["distance_range"] == pytest.approx(9.0)
    assert features["avg_direction_change"] == pytest.approx(90.0)
    assert features["current_distance"] == pytest.approx(10.0)
    assert features["current_bearing"] == pytest.approx(180.0)
    assert features["time_hours"] == pytest.approx(3.0)
    assert features["num_updates"] == 3
    assert features["log_growth_acres_24h"] == pytest.approx(4.976733742, rel=1e-6)


def test_anomaly_feature_math_exact_values():
    detector = AnomalyDetector()
    features = detector.extract_fire_behavior_features(
        make_fire(acres=500.0, contained=25.0),
        build_history(),
    )

    assert features["mean_spread_rate"] == pytest.approx(3.0)
    assert features["std_spread_rate"] == pytest.approx(0.0)
    assert features["mean_direction_change"] == pytest.approx(67.5)
    assert features["spread_acceleration"] == pytest.approx(0.0)
    assert features["containment_pct"] == pytest.approx(25.0)
    assert features["acreage"] == pytest.approx(500.0)
    assert features["current_distance"] == pytest.approx(10.0)


def test_risk_scorer_formula_and_trajectory_alignment():
    scorer = RiskScorer()
    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    fire = make_fire(acres=50.0, contained=20.0, bearing=0.0)
    fire.discovered_at = base_time
    fire.last_updated = base_time + timedelta(hours=24)
    fire.latitude = 0.0
    fire.longitude = 0.0

    fire_risk = scorer.calculate_fire_risk_score(fire)
    # 50 acres -> 100*log10(51)/log10(10001) = 42.689; 0.35*42.689 + 0.40*80 + 0.25*66.667
    assert fire_risk == pytest.approx(63.6077434950)

    north_location = make_location(location_id="north", latitude=1.0, longitude=0.0)
    east_location = make_location(location_id="east", latitude=0.0, longitude=1.0)
    south_location = make_location(location_id="south", latitude=-1.0, longitude=0.0)

    aligned = scorer._calculate_trajectory_risk(north_location, fire, 10.0, 5.0)
    crosswind = scorer._calculate_trajectory_risk(east_location, fire, 10.0, 5.0)
    opposite = scorer._calculate_trajectory_risk(south_location, fire, 10.0, 5.0)

    assert aligned == pytest.approx(97.2222222222)
    assert crosswind == pytest.approx(13.1570414473, rel=1e-3)
    assert opposite == pytest.approx(0.0326177552, rel=1e-3)
    assert aligned > crosswind > opposite


def test_advanced_weather_direction_and_fwi_math():
    service = AdvancedWeatherService()
    weather = WeatherData(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        temperature_f=95.0,
        humidity_pct=20.0,
        wind_speed_mph=20.0,
        wind_direction_deg=90,
        precipitation_in=0.0,
        pressure_mb=1013.25,
        dewpoint_f=50.0,
        visibility_miles=10.0,
        cloud_coverage_pct=0,
    )

    indices = service.calculate_fire_weather_index_v2(weather)

    # Fosberg by hand: EMC 4.0255%, damping 0.75743, sqrt(1 + 20^2) / 0.3002.
    assert indices["fwi"] == pytest.approx(50.528, abs=0.01)
    assert indices["danger_level"] == "HIGH"

    assert service._direction_to_degrees("N") == 0
    assert service._direction_to_degrees("SW") == 225
    assert service._direction_to_degrees("WNW") == 292
    assert service._direction_to_degrees("UNKNOWN") == 0


def test_comprehensive_data_distance_and_quality_math():
    los_angeles = (34.0522, -118.2437)
    san_francisco = (37.7749, -122.4194)

    distance = ComprehensiveDataService._haversine_distance(*los_angeles, *san_francisco)
    reverse_distance = ComprehensiveDataService._haversine_distance(*san_francisco, *los_angeles)
    fire_service_distance = FireDataService.haversine_distance(*los_angeles, *san_francisco)

    assert distance == pytest.approx(reverse_distance)
    assert distance == pytest.approx(fire_service_distance, rel=0.001)
    assert ComprehensiveDataService._haversine_distance(1.0, 2.0, 1.0, 2.0) == pytest.approx(0.0)

    completeness, confidence = ComprehensiveDataService._assess_data_quality(
        weather_data={
            "temperature_f": 95.0,
            "humidity_pct": 20.0,
            "wind_speed_mph": 20.0,
            "pressure_mb": 1013.25,
            "fwi": 3.85,
            "source": "NOAA",
        },
        satellite_data={
            "ndvi": 0.45,
            "lst_kelvin": 305.0,
            "cloud_coverage_pct": 20.0,
        },
        topo_data={
            "elevation_m": 500.0,
            "slope_deg": 15.0,
            "aspect_deg": 180.0,
        },
        veg_data={
            "fuel_model": "timber",
            "fuel_load_tons_acre": 12.0,
            "moisture_content_pct": 18.0,
        },
    )

    assert completeness == pytest.approx(100.0)
    assert confidence == pytest.approx(82.5)


def test_spread_model_benchmark_ranks_candidates_against_baseline():
    fire_samples = []
    for index in range(18):
        avg_spread_rate = 0.8 + (index * 0.28)
        distance_range = 1.2 + (index * 0.15)
        avg_direction_change = float((index % 6) * 12)
        current_distance = 10.0 - (index * 0.18)
        current_bearing = float((index * 23) % 360)
        time_hours = 1.0 + (index * 0.35)
        num_updates = 3 + (index % 5)

        spread_rate = (
            (0.68 * avg_spread_rate)
            + (0.11 * distance_range)
            + (0.025 * time_hours)
            + (0.015 * num_updates)
            - (0.0005 * avg_direction_change)
            + (0.02 if index % 2 == 0 else -0.015)
        )

        fire_samples.append(
            {
                "features": {
                    "avg_spread_rate": avg_spread_rate,
                    "distance_range": distance_range,
                    "avg_direction_change": avg_direction_change,
                    "current_distance": current_distance,
                    "current_bearing": current_bearing,
                    "time_hours": time_hours,
                    "num_updates": num_updates,
                },
                "spread_rate": spread_rate,
            }
        )

    benchmark = benchmark_spread_models(fire_samples, random_state=42, cv_splits=3)

    assert benchmark["best_model"] in {result["name"] for result in benchmark["results"]}
    assert len(benchmark["results"]) >= 4
    assert benchmark["baseline_model"] in {result["name"] for result in benchmark["results"]}
    assert benchmark["results"][0]["mae"] <= benchmark["results"][-1]["mae"]

    baseline = next(
        result for result in benchmark["results"] if result["name"] == benchmark["baseline_model"]
    )
    assert benchmark["results"][0]["mae"] <= baseline["mae"]
    assert benchmark["best_improvement_pct"] >= 0.0
    assert benchmark["dataset_is_informative"] is True
    assert benchmark["validation_warning"] is None


def test_spread_model_benchmark_flags_non_informative_dataset():
    fire_samples = [
        {
            "features": {
                "avg_spread_rate": 0.0,
                "distance_range": 0.0,
                "avg_direction_change": 0.0,
                "current_distance": 10.0 + index,
                "current_bearing": 90.0,
                "time_hours": 1.0 + (index * 0.1),
                "num_updates": 3 + (index % 2),
            },
            "spread_rate": 0.0,
            "metadata": {
                "fire_id": f"fire-{index // 2}",
                "location_id": f"loc-{index % 2}",
            },
        }
        for index in range(8)
    ]

    benchmark = benchmark_spread_models(fire_samples, random_state=42, cv_splits=3)

    assert benchmark["dataset_is_informative"] is False
    assert benchmark["target_unique_count"] == 1
    assert benchmark["target_std"] == pytest.approx(0.0)
    assert "cannot validate" in benchmark["validation_warning"].lower()


def test_algorithm_bakeoff_runs_and_reports_winner(monkeypatch):
    fire_samples = []
    for index in range(16):
        fire_samples.append(
            {
                "features": {
                    "avg_spread_rate": 0.5 + (index * 0.3),
                    "distance_range": 1.0 + (index * 0.2),
                    "avg_direction_change": float((index % 4) * 10),
                    "current_distance": 9.0 - (index * 0.15),
                    "current_bearing": float((index * 17) % 360),
                    "time_hours": 1.0 + (index * 0.25),
                    "num_updates": 3 + (index % 4),
                },
                "spread_rate": 0.7 + (index * 0.22),
                "metadata": {
                    "fire_id": f"fire-{index // 2}",
                    "location_id": f"loc-{index % 2}",
                },
            }
        )

    monkeypatch.setattr(
        "app.ml.benchmarking._build_optional_booster_models",
        lambda random_state: (
            {},
            ["xgboost_regressor", "lightgbm_regressor", "catboost_regressor"],
        ),
    )

    bakeoff = benchmark_algorithm_bakeoff(fire_samples, cv_splits=3, random_state=42)

    assert bakeoff["bakeoff_ran"] is True
    assert bakeoff["winner_model"] in {row["name"] for row in bakeoff["results"]}
    assert bakeoff["dataset_is_informative"] is True
    assert "xgboost_regressor" in bakeoff["unavailable_optional_models"]


def test_algorithm_bakeoff_skips_non_informative_labels():
    fire_samples = [
        {
            "features": {
                "avg_spread_rate": 0.0,
                "distance_range": 0.0,
                "avg_direction_change": 0.0,
                "current_distance": 8.0 + index,
                "current_bearing": 90.0,
                "time_hours": 1.0 + (index * 0.1),
                "num_updates": 4,
            },
            "spread_rate": 0.0,
            "metadata": {
                "fire_id": f"fire-{index // 2}",
                "location_id": f"loc-{index % 2}",
            },
        }
        for index in range(8)
    ]

    bakeoff = benchmark_algorithm_bakeoff(fire_samples, cv_splits=3)

    assert bakeoff["bakeoff_ran"] is False
    assert "skipped" in bakeoff["reason"].lower()


def test_promotion_gate_allows_stable_improving_model():
    benchmark = {
        "best_model": "extra_trees_ensemble",
        "baseline_model": "random_forest_baseline",
        "best_improvement_pct": 6.4,
        "dataset_is_informative": True,
        "effective_tie_with_baseline": False,
        "group_count": 10,
        "results": [
            {
                "name": "random_forest_baseline",
                "mae": 0.82,
                "rmse": 1.18,
                "r2": 0.29,
                "mae_std": 0.22,
                "rmse_std": 0.18,
                "r2_std": 0.12,
                "folds_used": 5,
            },
            {
                "name": "extra_trees_ensemble",
                "mae": 0.767,
                "rmse": 1.09,
                "r2": 0.36,
                "mae_std": 0.14,
                "rmse_std": 0.16,
                "r2_std": 0.1,
                "folds_used": 5,
            },
        ],
        "time_split": {
            "time_split_ran": True,
            "best_improvement_pct": 4.1,
            "best_model_metrics": {
                "train_mae": 0.61,
                "test_mae": 0.79,
                "test_r2": 0.14,
                "generalization_gap_pct": 29.5,
            },
        },
    }

    gate = evaluate_model_promotion_gate(
        benchmark,
        min_improvement_pct=2.0,
        require_informative=True,
        min_group_count=4,
        min_r2=0.1,
        max_mae_std_ratio=0.5,
        min_stability_score=0.6,
    )

    assert gate["promote"] is True
    assert gate["selected_model"] == "extra_trees_ensemble"
    assert gate["scorecard"]["production_ready"] is True


def test_promotion_gate_blocks_unstable_model():
    benchmark = {
        "best_model": "extra_trees_ensemble",
        "baseline_model": "random_forest_baseline",
        "best_improvement_pct": 4.0,
        "dataset_is_informative": True,
        "group_count": 8,
        "results": [
            {
                "name": "random_forest_baseline",
                "mae": 1.0,
                "rmse": 1.3,
                "r2": 0.21,
                "mae_std": 0.2,
                "rmse_std": 0.24,
                "r2_std": 0.08,
                "folds_used": 5,
            },
            {
                "name": "extra_trees_ensemble",
                "mae": 0.96,
                "rmse": 1.29,
                "r2": 0.24,
                "mae_std": 1.0,
                "rmse_std": 0.5,
                "r2_std": 0.3,
                "folds_used": 5,
            },
        ],
    }

    gate = evaluate_model_promotion_gate(
        benchmark,
        min_improvement_pct=2.0,
        require_informative=True,
        min_group_count=4,
        min_r2=0.1,
        max_mae_std_ratio=0.5,
        min_stability_score=0.6,
    )

    assert gate["promote"] is False
    assert "unstable" in gate["reason"].lower() or "stability" in gate["reason"].lower()


def test_production_readiness_scorecard_flags_non_sota_holdout_profile():
    scorecard = build_production_readiness_scorecard(
        {
            "can_benchmark": True,
            "dataset_is_informative": True,
            "best_improvement_pct": 1.1,
            "best_model_metrics": {"r2": 0.15},
            "time_split": {
                "time_split_ran": True,
                "best_improvement_pct": 0.58,
                "best_model_metrics": {
                    "test_r2": -0.0018,
                    "generalization_gap_pct": 394.8,
                },
                "drift_diagnosis": "High drift risk between historical train and holdout periods",
            },
        }
    )

    assert scorecard["grade"] == "C"
    assert scorecard["production_ready"] is False
    assert scorecard["state_of_the_art"] is False
    assert scorecard["assessment"].startswith("No, not yet")


def test_build_historical_spread_samples_from_distance_history(db):
    location = make_location(location_id="hist-loc", latitude=34.0, longitude=-118.0)
    fire = make_fire(acres=900.0, contained=15.0)
    fire.fire_id = "hist-fire"
    db.add_all([location, fire])
    db.commit()
    db.refresh(location)
    db.refresh(fire)

    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    distances = [10.0, 8.6, 7.2, 5.9, 4.8, 3.9, 3.1, 2.4]
    for index, distance in enumerate(distances):
        db.add(
            DistanceHistory(
                fire_id=fire.id,
                location_id=location.id,
                distance_miles=distance,
                bearing_degrees=90.0,
                direction="E",
                moving_towards=True,
                speed_mph=1.0,
                timestamp=base_time + timedelta(hours=index),
                observed_acres_burned=900.0 + (index * 55.0),
                observed_percent_contained=max(0.0, 15.0 - index),
                observed_growth_acres_24h=132.0,
            )
        )
    db.commit()

    samples = build_historical_spread_samples(db)

    assert len(samples) >= 6
    assert all(sample["spread_rate"] >= 0.0 for sample in samples)
    assert {"avg_spread_rate", "distance_range", "num_updates", "log_growth_acres_24h"}.issubset(
        samples[0]["features"]
    )
    assert samples[0]["features"]["log_growth_acres_24h"] > 0.0
    assert samples[0]["metadata"]["label_signal"] in {"distance", "area_growth", "growth_hint"}


def test_build_historical_spread_samples_uses_area_growth_when_distance_is_flat(db):
    location = make_location(location_id="hist-growth-loc", latitude=34.0, longitude=-118.0)
    fire = make_fire(acres=400.0, contained=10.0)
    fire.fire_id = "hist-growth-fire"
    db.add_all([location, fire])
    db.commit()
    db.refresh(location)
    db.refresh(fire)

    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    observations = [
        (6.0, 400.0, 0.0),
        (5.99, 640.0, 240.0),
        (5.98, 900.0, 260.0),
        (5.97, 1225.0, 325.0),
    ]
    for index, (distance, acres, growth) in enumerate(observations):
        db.add(
            DistanceHistory(
                fire_id=fire.id,
                location_id=location.id,
                distance_miles=distance,
                bearing_degrees=90.0,
                direction="E",
                moving_towards=True,
                speed_mph=0.0,
                timestamp=base_time + timedelta(hours=index),
                observed_acres_burned=acres,
                observed_percent_contained=10.0,
                observed_growth_acres_24h=growth,
            )
        )
    db.commit()

    samples = build_historical_spread_samples(db, min_distance_delta_miles=0.05)

    assert samples
    assert any(
        sample["metadata"]["label_signal"] in {"area_growth", "growth_hint"} for sample in samples
    )
    assert all(sample["spread_rate"] > 0.0 for sample in samples)


def test_fire_risk_keeps_rising_with_size_up_to_large_fires():
    from types import SimpleNamespace

    scorer = RiskScorer()

    def risk(acres):
        fire = SimpleNamespace(acres_burned=acres, percent_contained=50.0, discovered_at=None)
        return scorer.calculate_fire_risk_score(fire)

    sizes = [1, 10, 100, 1_000, 10_000]
    scores = [risk(a) for a in sizes]
    assert scores == sorted(scores) and len(set(scores)) == len(scores)
    assert risk(10_000) == risk(1_000_000)
