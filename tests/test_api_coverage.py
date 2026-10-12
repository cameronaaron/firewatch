import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import alerts as alerts_api
from app.api import fires as fires_api
from app.api import locations as locations_api
from app.api import predictions as predictions_api
from app.models.database import Alert, DistanceHistory, Fire, Location
from app.schemas import AlertLevelEnum, LocationUpdate


def make_location(location_id="loc-1", latitude=34.0, longitude=-118.0, active=True):
    return Location(
        location_id=location_id,
        address="123 Test St",
        display_name="Test Location",
        icon="🏠",
        latitude=latitude,
        longitude=longitude,
        is_active=active,
    )


def make_fire(fire_id="fire-1", active=True):
    fire = Fire(
        fire_id=fire_id,
        name="Test Fire",
        latitude=34.1,
        longitude=-118.1,
        acres_burned=5000,
        percent_contained=10,
        is_active=active,
    )
    fire.bearing_degrees = 90
    return fire


def seed_alert_graph(db):
    location = make_location("watch-home")
    fire = make_fire("near-fire")
    db.add_all([location, fire])
    db.commit()
    db.refresh(location)
    db.refresh(fire)

    history = DistanceHistory(
        fire_id=fire.id,
        location_id=location.id,
        distance_miles=1.5,
        bearing_degrees=90.0,
        direction="E",
        moving_towards=True,
        speed_mph=3.0,
        timestamp=datetime.now(UTC),
    )
    alert = Alert(
        fire_id=fire.id,
        location_id=location.id,
        alert_level="WARNING",
        distance_miles=1.5,
        reason="Fire nearby",
        timestamp=datetime.now(UTC),
        acknowledged=False,
    )
    db.add_all([history, alert])
    db.commit()
    db.refresh(history)
    db.refresh(alert)
    return location, fire, history, alert


def test_locations_update_and_delete(db, monkeypatch):
    location = make_location("edit-me")
    db.add(location)
    db.commit()

    async def fake_geocode(address):
        return 40.0, -120.0

    monkeypatch.setattr(
        locations_api.GeocodingService, "geocode_location", staticmethod(fake_geocode)
    )
    updated = asyncio.run(
        locations_api.update_location(
            "edit-me",
            LocationUpdate(
                address="456 Updated St",
                display_name="Updated",
                icon="🔥",
                is_active=False,
            ),
            db,
        )
    )
    assert updated.address is None
    assert updated.latitude == 40.0
    assert updated.is_active is False

    async def broken_geocode(address):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        locations_api.GeocodingService, "geocode_location", staticmethod(broken_geocode)
    )
    with pytest.raises(HTTPException):
        asyncio.run(
            locations_api.update_location(
                "edit-me",
                LocationUpdate(address="789 Broken St"),
                db,
            )
        )

    updated = asyncio.run(
        locations_api.update_location(
            "edit-me",
            LocationUpdate(latitude=41.0, longitude=-121.0),
            db,
        )
    )
    assert updated.address is None
    assert updated.latitude == 41.0

    with pytest.raises(HTTPException):
        asyncio.run(locations_api.update_location("missing", LocationUpdate(display_name="x"), db))

    assert (
        locations_api.delete_location("edit-me", db)["message"] == "Location deleted successfully"
    )
    with pytest.raises(HTTPException):
        locations_api.delete_location("missing", db)


def test_alerts_routes_cover_filters_and_assessment(db):
    location, fire, history, alert = seed_alert_graph(db)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        alerts_api.smoke_service, "get_smoke_assessment", lambda latitude, longitude: None
    )
    monkeypatch.setattr(
        alerts_api.air_quality_service,
        "get_air_quality_assessment",
        lambda latitude, longitude: None,
    )

    all_alerts = alerts_api.list_alerts(
        location_id=None, alert_level=None, acknowledged=None, limit=50, db=db
    )
    assert len(all_alerts) == 1
    filtered = alerts_api.list_alerts(
        location_id=location.location_id,
        alert_level=AlertLevelEnum.WARNING,
        acknowledged=False,
        limit=50,
        db=db,
    )
    assert len(filtered) == 1

    with pytest.raises(HTTPException):
        alerts_api.list_alerts(
            location_id="missing", alert_level=None, acknowledged=None, limit=50, db=db
        )

    assert alerts_api.get_alert(alert.id, db).id == alert.id
    with pytest.raises(HTTPException):
        alerts_api.get_alert(99999, db)

    assert alerts_api.acknowledge_alert(alert.id, db)["message"] == "Alert acknowledged"
    db.refresh(alert)
    assert alert.acknowledged is True
    with pytest.raises(HTTPException):
        alerts_api.acknowledge_alert(99999, db)

    safe_location = make_location("safe-home", latitude=0.0, longitude=0.0)
    db.add(safe_location)
    db.commit()
    safe_assessment = alerts_api.get_threat_assessment("safe-home", db)
    assert safe_assessment.alert_level == AlertLevelEnum.SAFE

    ungeocoded = make_location("ungeocoded", None, None)
    db.add(ungeocoded)
    db.commit()
    with pytest.raises(HTTPException):
        alerts_api.get_threat_assessment("ungeocoded", db)

    assessment = alerts_api.get_threat_assessment(location.location_id, db)
    assert assessment.nearest_fire_name == fire.name
    assert assessment.alert_level in AlertLevelEnum
    assert alerts_api._get_recommendations("UNKNOWN", fire, history) == [
        "Data unknown; remain on alert"
    ]

    with pytest.raises(HTTPException):
        alerts_api.get_threat_assessment("missing", db)

    monkeypatch.undo()


def test_alerts_assessment_uses_smoke_signal_when_no_fire(db, monkeypatch):
    location = make_location("smoke-home")
    db.add(location)
    db.commit()

    monkeypatch.setattr(
        alerts_api.smoke_service,
        "get_smoke_assessment",
        lambda latitude, longitude: {
            "smoke_present": True,
            "smoke_density": "Heavy",
            "smoke_distance_miles": 0.0,
            "smoke_scope": "overhead",
            "smoke_source": "NOAA_HMS",
            "smoke_observed_at": datetime.now(UTC),
        },
    )
    monkeypatch.setattr(
        alerts_api.air_quality_service,
        "get_air_quality_assessment",
        lambda latitude, longitude: None,
    )

    assessment = alerts_api.get_threat_assessment(location.location_id, db)

    assert assessment.alert_level == AlertLevelEnum.MONITOR
    assert assessment.smoke_density == "Heavy"
    assert any("smoke" in recommendation.lower() for recommendation in assessment.recommendations)


def test_alerts_assessment_uses_live_fire_fallback_without_history(db, monkeypatch):
    location = make_location("new-watch", latitude=34.0, longitude=-118.0)
    fire = make_fire("live-fire")
    db.add_all([location, fire])
    db.commit()

    monkeypatch.setattr(
        alerts_api.smoke_service, "get_smoke_assessment", lambda latitude, longitude: None
    )
    monkeypatch.setattr(
        alerts_api.air_quality_service,
        "get_air_quality_assessment",
        lambda latitude, longitude: None,
    )

    assessment = alerts_api.get_threat_assessment(location.location_id, db)

    assert assessment.nearest_fire_name == fire.name
    assert assessment.distance_miles is not None
    assert assessment.direction is not None
    assert assessment.speed_mph == 0.0
    assert assessment.moving_towards is False


def test_alerts_assessment_uses_air_quality_signal_when_no_fire(db, monkeypatch):
    location = make_location("aq-home")
    db.add(location)
    db.commit()

    monkeypatch.setattr(
        alerts_api.smoke_service, "get_smoke_assessment", lambda latitude, longitude: None
    )
    monkeypatch.setattr(
        alerts_api.air_quality_service,
        "get_air_quality_assessment",
        lambda latitude, longitude: {
            "air_quality_index": 112,
            "pm25_ug_m3": 41.2,
            "air_quality_category": "Unhealthy for Sensitive Groups",
            "air_quality_source": "Open-Meteo Air Quality",
            "air_quality_observed_at": datetime.now(UTC),
        },
    )

    assessment = alerts_api.get_threat_assessment(location.location_id, db)

    assert assessment.alert_level == AlertLevelEnum.PREPARE
    assert assessment.air_quality_index == 112
    assert assessment.air_quality_category == "Unhealthy for Sensitive Groups"
    assert any(
        "aqi" in recommendation.lower() or "air quality" in recommendation.lower()
        for recommendation in assessment.recommendations
    )


def test_fires_routes_cover_success_and_not_found(db):
    location, fire, history, alert = seed_alert_graph(db)
    inactive_fire = make_fire("inactive-fire", active=False)
    db.add(inactive_fire)
    db.commit()

    active_fires = fires_api.list_fires(active_only=True, limit=100, db=db)
    assert [item.fire_id for item in active_fires] == [fire.fire_id]
    assert len(fires_api.list_fires(active_only=False, limit=100, db=db)) == 2

    assert fires_api.get_fire(fire.fire_id, db).fire_id == fire.fire_id
    with pytest.raises(HTTPException):
        fires_api.get_fire("missing", db)

    recent_fire_alerts = fires_api.get_fire_alerts(fire.fire_id, hours=24, db=db)
    assert len(recent_fire_alerts) == 1
    with pytest.raises(HTTPException):
        fires_api.get_fire_alerts("missing", db=db)

    recent = fires_api.get_recent_alerts(hours=24, limit=10, db=db)
    assert len(recent) == 1


def test_prediction_threat_evacuation_and_forecast_paths(db, monkeypatch):
    with pytest.raises(HTTPException):
        asyncio.run(predictions_api.get_threat_prediction("missing", db))

    location = make_location("pred-home", None, None)
    db.add(location)
    db.commit()
    with pytest.raises(HTTPException):
        asyncio.run(predictions_api.get_threat_prediction("pred-home", db))

    location.latitude = 34.0
    location.longitude = -118.0
    db.commit()
    safe = asyncio.run(predictions_api.get_threat_prediction("pred-home", db))
    assert safe["status"] == "safe"

    fire = make_fire("pred-fire")
    db.add(fire)
    db.commit()

    async def fake_analyze_fire_threat(fire_obj, location_obj, distance, db_session):
        return {
            "threat_score": 77,
            "threat_level": "Very High",
            "fire_risk": 80,
            "location_vulnerability": 70,
            "predictions": {
                "spread_rate_mph": 2.0,
                "adjusted_spread_rate_mph": 3.0,
                "impact_timeline": {"hours_until_impact": 4},
                "containment_forecast": None,
                "direction_probabilities": {0: 1.0},
            },
            "weather": {
                "current": None,
                "fire_weather_index": 50,
                "fire_danger_level": "HIGH",
                "spread_multiplier": 1.5,
            },
            "anomalies": {"is_anomalous": False, "reasons": [], "insight": None},
            "recommendations": ["Leave soon"],
        }

    async def fake_evacuation(location_obj, fires, db_session):
        return {
            "urgency_level": "WARNING",
            "timing": "Soon",
            "primary_threat": {"fire_id": fire.fire_id},
            "threat_ranking": [],
        }

    async def fake_forecast(location_obj, forecast_hours):
        return {"location_id": location_obj.location_id, "predictions": [{"date": "2026-05-21"}]}

    monkeypatch.setattr(predictions_api.ml_service, "analyze_fire_threat", fake_analyze_fire_threat)
    monkeypatch.setattr(predictions_api.ml_service, "get_evacuation_optimization", fake_evacuation)
    monkeypatch.setattr(predictions_api.ml_service, "get_predictive_analytics", fake_forecast)

    threat = asyncio.run(predictions_api.get_threat_prediction("pred-home", db))
    assert threat["threat_score"] == 77

    with pytest.raises(HTTPException):
        asyncio.run(predictions_api.get_evacuation_plan("missing", db))
    location.latitude = None
    location.longitude = None
    db.commit()
    with pytest.raises(HTTPException):
        asyncio.run(predictions_api.get_evacuation_plan("pred-home", db))
    location.latitude = 34.0
    location.longitude = -118.0
    db.commit()
    evacuation = asyncio.run(predictions_api.get_evacuation_plan("pred-home", db))
    assert evacuation["urgency_level"] == "WARNING"

    with pytest.raises(HTTPException):
        asyncio.run(predictions_api.get_predictive_forecast("missing", 24, db))
    location.latitude = None
    location.longitude = None
    db.commit()
    with pytest.raises(HTTPException):
        asyncio.run(predictions_api.get_predictive_forecast("pred-home", 24, db))
    location.latitude = 34.0
    location.longitude = -118.0
    db.commit()
    forecast = asyncio.run(predictions_api.get_predictive_forecast("pred-home", 48, db))
    assert forecast["location_id"] == "pred-home"


def test_prediction_spread_anomaly_timeline_risk_and_weather_paths(db, monkeypatch):
    location = make_location("risk-home")
    fire = make_fire("risk-fire")
    db.add_all([location, fire])
    db.commit()

    predictor = predictions_api.ml_service.predictor
    detector = predictions_api.ml_service.anomaly_detector
    scorer = predictions_api.ml_service.risk_scorer
    weather_service = predictions_api.ml_service.weather_service

    monkeypatch.setattr(predictor, "predict_spread_rate", lambda fire_obj, history: 2.5)
    monkeypatch.setattr(
        detector,
        "detect_anomaly",
        lambda fire_obj, history: {"is_anomaly": True, "anomaly_score": -0.8, "reasons": ["fast"]},
    )
    monkeypatch.setattr(detector, "get_anomaly_insight", lambda fire_obj, history: "fast")
    monkeypatch.setattr(
        predictor,
        "predict_impact_timeline",
        lambda fire_obj, history, target_distance: {"hours_until_impact": 3},
    )
    monkeypatch.setattr(scorer, "calculate_fire_risk_score", lambda fire_obj: 80)
    monkeypatch.setattr(
        scorer,
        "calculate_location_vulnerability_score",
        lambda location_obj, fire_obj, distance: 65,
    )
    monkeypatch.setattr(
        scorer, "calculate_combined_threat_score", lambda location_obj, fire_obj, distance: 72
    )
    monkeypatch.setattr(scorer, "get_threat_description", lambda score: "High")
    monkeypatch.setattr(scorer, "get_recommendations", lambda score: ["Prepare"])

    async def fake_current_weather(latitude, longitude):
        return {"temperature_f": 90}

    async def fake_fwi(latitude, longitude):
        return {"fwi_index": 60, "danger_level": "HIGH"}

    async def fake_weather_forecast(latitude, longitude, days=3):
        return {"forecast": [{"date": "2026-05-21"}]}

    monkeypatch.setattr(weather_service, "get_current_weather", fake_current_weather)
    monkeypatch.setattr(weather_service, "get_fire_weather_indices", fake_fwi)
    monkeypatch.setattr(weather_service, "get_forecast", fake_weather_forecast)

    with pytest.raises(HTTPException):
        predictions_api.get_fire_spread_rate_prediction("missing", db=db)
    spread_rate = predictions_api.get_fire_spread_rate_prediction(fire.fire_id, db=db)
    assert spread_rate["confidence"] == 0.7

    class FakeQuery:
        def __init__(self, item):
            self.item = item

        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return self.item

    class FakeDB:
        def query(self, model):
            return FakeQuery(SimpleNamespace(fire_id="placeholder"))

    fallback_spread = predictions_api.get_fire_spread_rate_prediction("placeholder", db=FakeDB())
    assert fallback_spread["confidence"] == 0.5

    with pytest.raises(HTTPException):
        predictions_api.detect_fire_anomalies("missing", db=db)
    anomalies = predictions_api.detect_fire_anomalies(fire.fire_id, db=db)
    assert anomalies["is_anomalous"] is True

    with pytest.raises(HTTPException):
        predictions_api.get_fire_impact_timeline("missing", location.location_id, db=db)
    with pytest.raises(HTTPException):
        predictions_api.get_fire_impact_timeline(fire.fire_id, "missing", db=db)
    timeline = predictions_api.get_fire_impact_timeline(fire.fire_id, location.location_id, db=db)
    assert timeline["impact_prediction"]["hours_until_impact"] == 3

    with pytest.raises(HTTPException):
        predictions_api.calculate_risk_score("missing", fire.fire_id, db=db)
    with pytest.raises(HTTPException):
        predictions_api.calculate_risk_score(location.location_id, "missing", db=db)
    risk_score = predictions_api.calculate_risk_score(
        location.location_id, fire.fire_id, distance_miles=5.0, db=db
    )
    assert risk_score["combined_threat_score"] == 72.0

    with pytest.raises(HTTPException):
        asyncio.run(predictions_api.get_weather_impact("missing", db))
    location.latitude = None
    location.longitude = None
    db.commit()
    with pytest.raises(HTTPException):
        asyncio.run(predictions_api.get_weather_impact(location.location_id, db))
    location.latitude = 34.0
    location.longitude = -118.0
    db.commit()
    weather = asyncio.run(predictions_api.get_weather_impact(location.location_id, db))
    assert weather["fire_danger_level"] == "HIGH"
