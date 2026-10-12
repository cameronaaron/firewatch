import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.core import database as core_database
from app.core.exceptions import FireDataFetchError
from app.main import app as fastapi_app
from app.main import fetch_fires_periodically, lifespan
from app.ml.anomaly_detection import AnomalyDetector
from app.ml.fire_prediction import FireSpreadPredictor
from app.ml.risk_scoring import RiskScorer
from app.models.database import Fire, Location
from app.services.fire_data import FireDataService


def make_location(location_id="loc", latitude=34.0, longitude=-118.0):
    return Location(
        location_id=location_id,
        address="123 Test St",
        display_name="Location",
        icon="📍",
        latitude=latitude,
        longitude=longitude,
        is_active=True,
    )


def make_fire(fire_id="fire", acres=1000, contained=50):
    fire = Fire(
        fire_id=fire_id,
        name="Fire",
        latitude=34.1,
        longitude=-118.1,
        acres_burned=acres,
        percent_contained=contained,
    )
    fire.bearing_degrees = 90
    return fire


def test_init_db_error_branch(monkeypatch):
    monkeypatch.setattr(
        core_database.Base.metadata,
        "create_all",
        lambda bind: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError):
        core_database.init_db()


def test_fetch_periodic_no_locations_and_existing_fire_update(db, monkeypatch):
    class StopLoop(BaseException):
        pass

    async def stop_sleep(seconds):
        raise StopLoop()

    monkeypatch.setattr("app.main.asyncio.sleep", stop_sleep)

    with pytest.raises(StopLoop):
        asyncio.run(fetch_fires_periodically(db))

    location = make_location("watch-home")
    fire = make_fire("existing-fire", acres=100, contained=90)
    db.add_all([location, fire])
    db.commit()

    async def fake_process(db_session, locations):
        return [
            {
                "fire_id": "existing-fire",
                "name": "Updated Fire",
                "latitude": 35.0,
                "longitude": -119.0,
                "acres_burned": 9999,
                "percent_contained": 5,
                "distances": {"watch-home": 2.0},
                "bearings": {"watch-home": 45.0},
                "directions": {"watch-home": "NE"},
                "movement_data": {"watch-home": {"moving_towards": False, "speed": 1.0}},
            }
        ]

    monkeypatch.setattr("app.main.FireDataService.process_fire_data", staticmethod(fake_process))
    with pytest.raises(StopLoop):
        asyncio.run(fetch_fires_periodically(db))

    db.refresh(fire)
    assert fire.name == "Updated Fire"
    assert fire.latitude == 35.0
    assert fire.percent_contained == 5


def test_lifespan_warning_branches(monkeypatch):
    class FakeQuery:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return None

    class FailingInitSession:
        def query(self, model):
            return FakeQuery()

        def add(self, obj):
            return None

        def commit(self):
            raise RuntimeError("init failed")

        def close(self):
            return None

    class PassiveSession(FailingInitSession):
        def commit(self):
            return None  # pragma: no cover

    sessions = [FailingInitSession(), PassiveSession(), PassiveSession()]

    def fake_session_local():
        return sessions.pop(0)

    class FakeTask:
        def __init__(self, coro):
            coro.close()
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

        def __await__(self):
            async def waiter():
                if self.cancelled:
                    raise asyncio.CancelledError()
                return None  # pragma: no cover

            return waiter().__await__()  # pragma: no cover

    async def broken_geocode(db):
        raise RuntimeError("geocode failed")

    monkeypatch.setattr("app.main.init_db", lambda: None)
    monkeypatch.setattr("app.core.database.SessionLocal", fake_session_local)
    monkeypatch.setattr("app.main.GeocodingService.geocode_locations", staticmethod(broken_geocode))
    monkeypatch.setattr("app.main.asyncio.create_task", lambda coro: FakeTask(coro))
    monkeypatch.setattr(
        "app.main.settings.MONITORED_LOCATIONS",
        {"demo": {"address": "x", "display_name": "Demo", "icon": "📍"}},
    )

    async def run_once():
        async with lifespan(fastapi_app):
            return None

    asyncio.run(run_once())


def test_anomaly_detector_residual_branches(monkeypatch):
    detector = AnomalyDetector()
    fire = make_fire(acres=200000, contained=0)

    detector.train([])
    default_message = detector.get_anomaly_insight(
        fire,
        [],
    )
    assert default_message is None

    obvious_detector = AnomalyDetector()
    obvious_detector.extract_fire_behavior_features = lambda fire_obj, history: {
        "mean_spread_rate": 25.0,
        "std_spread_rate": 0.0,
        "mean_direction_change": 50.0,
        "spread_acceleration": 11.0,
        "containment_pct": 101.0,
        "acreage": 1000.0,
        "current_distance": 5.0,
    }
    obvious_result = obvious_detector.detect_anomaly(fire, [])
    assert obvious_result["is_anomaly"] is True
    assert len(obvious_result["reasons"]) == 4

    detector.extract_fire_behavior_features = lambda fire_obj, history: {
        "mean_spread_rate": 11.0,
        "std_spread_rate": 0.5,
        "mean_direction_change": 31.0,
        "spread_acceleration": 6.0,
        "containment_pct": 0.0,
        "acreage": 100001.0,
        "current_distance": 1.0,
    }
    detector._check_obvious_anomalies = lambda features: []
    detector.is_trained = True
    detector.model.predict = lambda rows: [-1]
    detector.model.score_samples = lambda rows: [-0.5]

    result = detector.detect_anomaly(fire, [])
    assert result["is_anomaly"] is True
    assert result["reasons"]

    broken_detector = AnomalyDetector()
    broken_detector.extract_fire_behavior_features = detector.extract_fire_behavior_features
    broken_detector._check_obvious_anomalies = lambda features: []
    broken_detector.is_trained = True
    broken_detector.model.predict = lambda rows: (_ for _ in ()).throw(RuntimeError("broken model"))
    broken_result = broken_detector.detect_anomaly(fire, [])
    assert broken_result["is_anomaly"] is False

    obvious = AnomalyDetector()._check_obvious_anomalies(
        {
            "mean_spread_rate": 25.0,
            "std_spread_rate": 0.0,
            "mean_direction_change": 50.0,
            "spread_acceleration": 11.0,
            "containment_pct": 101.0,
            "acreage": 1000.0,
            "current_distance": 5.0,
        }
    )
    assert len(obvious) == 4

    explainer = AnomalyDetector()
    reasons = explainer._explain_anomaly(
        {
            "mean_spread_rate": -2.0,
            "std_spread_rate": 0.0,
            "mean_direction_change": 40.0,
            "spread_acceleration": -6.0,
            "containment_pct": 0.0,
            "acreage": 150000.0,
            "current_distance": 1.0,
        }
    )
    assert "Fire distance decreasing unusually fast" in reasons
    assert "Spread rate slowing unexpectedly" in reasons

    explainer.detect_anomaly = lambda fire_obj, history: {"is_anomaly": True, "reasons": []}
    assert (
        explainer.get_anomaly_insight(fire, [])
        == "Fire behavior deviates from normal patterns - monitoring advised"
    )


def test_fire_prediction_risk_and_fire_data_residual_branches(db, monkeypatch):
    predictor = FireSpreadPredictor()
    predictor.train([])
    fire = make_fire(contained=50)
    history = [
        SimpleNamespace(
            timestamp=datetime.now(UTC) - timedelta(hours=1),
            distance_miles=1.0,
            bearing_degrees=0.0,
        ),
        SimpleNamespace(timestamp=datetime.now(UTC), distance_miles=2.0, bearing_degrees=90.0),
    ]

    predictor.is_trained = True
    predictor.spread_model.predict = lambda rows: (_ for _ in ()).throw(
        RuntimeError("predict failed")
    )
    assert predictor.predict_spread_rate(fire, history) >= 0.0
    assert predictor.predict_impact_timeline(fire, [], target_distance=1.0) is None

    immediate = predictor.predict_impact_timeline(fire, history, target_distance=5.0)
    assert immediate["hours_until_impact"] == 0.0

    same_time = datetime.now(UTC)
    zero_time_history = [
        SimpleNamespace(timestamp=same_time, distance_miles=4.0, bearing_degrees=0.0),
        SimpleNamespace(timestamp=same_time, distance_miles=3.0, bearing_degrees=10.0),
    ]
    assert predictor.predict_containment_time(fire, zero_time_history) is None

    negative_fire = make_fire(contained=-10)
    assert predictor.predict_containment_time(negative_fire, history) is None
    contained_fire = make_fire(contained=99)
    assert predictor.predict_containment_time(contained_fire, history)["confidence"] == 0.95

    scorer = RiskScorer()
    recent_fire = make_fire(contained=25)
    recent_fire.discovered_at = datetime.now(UTC) - timedelta(hours=12)
    recent_fire.last_updated = datetime.now(UTC)
    assert scorer.calculate_fire_risk_score(recent_fire) > 0

    assert scorer._calculate_trajectory_risk(make_location(), SimpleNamespace(), 10.0, 1.0) == 0.0
    assert scorer.get_recommendations(5)
    assert scorer.get_recommendations(35)
    assert scorer.get_recommendations(55)
    assert scorer.get_recommendations(75)

    async def raw_fetch_skip():
        return [
            {"Name": "Bad", "UniqueId": "bad", "Latitude": "not-a-float", "Longitude": 0},
            {"Name": "Far", "UniqueId": "far", "Latitude": 0.0, "Longitude": 0.0},
        ]

    async def no_firms(cls, locations):
        return []

    monkeypatch.setattr(FireDataService, "_fetch_firms_near_locations", classmethod(no_firms))
    monkeypatch.setattr(FireDataService, "fetch_fires_from_api", staticmethod(raw_fetch_skip))
    result = asyncio.run(
        FireDataService.process_fire_data(
            db, [make_location("nocoords", None, None), make_location("coords", 34.0, -118.0)]
        )
    )
    assert result == []

    async def raw_fetch_far_distance():
        return [
            {"Name": "Farther", "UniqueId": "farther", "Latitude": 40.0, "Longitude": -100.0},
        ]

    monkeypatch.setattr("app.services.fire_data.settings.MAX_FIRE_DISTANCE_MILES", 0.1)
    monkeypatch.setattr(
        FireDataService, "fetch_fires_from_api", staticmethod(raw_fetch_far_distance)
    )
    result = asyncio.run(
        FireDataService.process_fire_data(
            db, [make_location("nocoords-2", None, None), make_location("coords-2", 34.0, -118.0)]
        )
    )
    assert result == []

    async def raw_fetch_error():
        raise FireDataFetchError("boom")

    monkeypatch.setattr(FireDataService, "fetch_fires_from_api", staticmethod(raw_fetch_error))
    with pytest.raises(FireDataFetchError):
        asyncio.run(FireDataService.process_fire_data(db, []))

    FireDataService._distance_history_cache.clear()
    FireDataService.calculate_movement("fire-x", "loc-x", 10.0)
    FireDataService._distance_history_cache["fire-x_loc-x"][0]["timestamp"] = datetime.now(
        UTC
    ) + timedelta(hours=1)
    moving_towards, speed = FireDataService.calculate_movement("fire-x", "loc-x", 9.0)
    assert moving_towards is False and speed == 0.0
    assert FireDataService.get_alert_level(None) == "UNKNOWN"
    assert FireDataService.get_alert_level(2.5, acres_burned=0, percent_contained=100) == "PREPARE"
    assert (
        FireDataService.get_alert_level(1_000_000.0, acres_burned=0, percent_contained=100)
        == "SAFE"
    )
