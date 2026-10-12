import asyncio
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.exceptions import RequestValidationError
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from starlette.requests import Request

from app.core import database as core_database
from app.core.config import get_settings
from app.core.exceptions import (
    DatabaseError,
    FireDataFetchError,
    FirewatchException,
    GeocodeError,
    LocationNotFoundError,
    ValidationError,
    WeatherFetchError,
)
from app.core.logging_config import JSONFormatter, setup_logging
from app.main import (
    app as fastapi_app,
)
from app.main import (
    fetch_fires_periodically,
    firewatch_exception_handler,
    lifespan,
    root,
    validation_exception_handler,
)
from app.ml.anomaly_detection import AnomalyDetector
from app.ml.fire_prediction import FireSpreadPredictor
from app.models.database import Alert, DistanceHistory, Fire, Location, WeatherCache
from app.services.advanced_weather import AdvancedWeatherService, WeatherData
from app.services.air_quality import AirQualityService
from app.services.comprehensive_data import ComprehensiveDataService
from app.services.fire_data import FireDataService
from app.services.geocoding import GeocodingService
from app.services.ml_analysis import MLFireAnalysisService
from app.services.satellite_service import FireDetection, SatelliteFireService
from app.services.smoke_data import SmokeDataService


class FakeAsyncResponse:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return self.payload


class FakeAsyncClient:
    def __init__(self, payload=None, error=None, status_error=None):
        self.payload = payload
        self.error = error
        self.status_error = status_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, *args, **kwargs):
        if self.error:
            raise self.error
        return FakeAsyncResponse(self.payload, self.status_error)


def build_location(location_id="loc-1", latitude=34.0, longitude=-118.0):
    return Location(
        location_id=location_id,
        address="123 Test St",
        display_name="Test Location",
        icon="🏠",
        latitude=latitude,
        longitude=longitude,
        is_active=True,
    )


def build_fire(fire_id="fire-1", acres=5000, contained=20, latitude=34.1, longitude=-118.1):
    fire = Fire(
        fire_id=fire_id,
        name="Test Fire",
        latitude=latitude,
        longitude=longitude,
        acres_burned=acres,
        percent_contained=contained,
    )
    fire.bearing_degrees = 90
    return fire


def test_exception_codes_and_model_reprs():
    assert FirewatchException("boom").code == "FIREWATCH_ERROR"
    assert FireDataFetchError("boom").code == "FIRE_DATA_FETCH_ERROR"
    assert LocationNotFoundError("missing").code == "LOCATION_NOT_FOUND"
    assert GeocodeError("addr").code == "GEOCODE_ERROR"
    assert WeatherFetchError("boom").code == "WEATHER_FETCH_ERROR"
    assert ValidationError("bad", field="name").field == "name"
    assert DatabaseError("db").code == "DATABASE_ERROR"

    location = build_location()
    fire = build_fire()
    history = DistanceHistory(fire_id=1, location_id=2, distance_miles=3.5)
    alert = Alert(
        fire_id=1,
        location_id=2,
        alert_level="WARNING",
        distance_miles=1.2,
        reason="Too close",
    )
    weather_cache = WeatherCache(
        latitude=34.0,
        longitude=-118.0,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    assert "Location" in repr(location)
    assert "Fire" in repr(fire)
    assert "DistanceHistory" in repr(history)
    assert "Alert" in repr(alert)
    assert "WeatherCache" in repr(weather_cache)


def test_json_formatter_and_setup_logging(monkeypatch):
    formatter = JSONFormatter()

    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            name="firewatch.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=123,
            msg="structured message",
            args=(),
            exc_info=sys.exc_info(),
        )
        record.extra = {"request_id": "req-1"}

    data = json.loads(formatter.format(record))
    assert data["message"] == "structured message"
    assert data["request_id"] == "req-1"
    assert "exception" in data

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    root_logger.handlers.clear()
    try:
        setup_logging(level="DEBUG", log_format="json")
        setup_logging(level="INFO", log_format="text")
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("geopy").level == logging.WARNING
    finally:
        root_logger.handlers = original_handlers


def test_sqlite_column_migration_skips_missing_table_and_fills_old_one(monkeypatch):
    from sqlalchemy import create_engine, text

    engine = create_engine("sqlite://")
    monkeypatch.setattr(core_database, "engine", engine)

    core_database._ensure_sqlite_distance_history_columns()
    with engine.connect() as connection:
        assert not list(connection.execute(text("PRAGMA table_info(distance_history)")))

    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE distance_history (id INTEGER PRIMARY KEY)"))
    core_database._ensure_sqlite_distance_history_columns()
    with engine.connect() as connection:
        columns = {
            row[1] for row in connection.execute(text("PRAGMA table_info(distance_history)"))
        }
    assert {
        "observed_acres_burned",
        "observed_percent_contained",
        "observed_growth_acres_24h",
    } <= columns


def test_database_helpers(monkeypatch):
    created = []
    monkeypatch.setattr(
        core_database.Base.metadata,
        "create_all",
        lambda bind: created.append(bind),
    )
    core_database.init_db()
    assert created == [core_database.engine]

    class FakeSession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_session = FakeSession()
    monkeypatch.setattr(core_database, "SessionLocal", lambda: fake_session)
    generator = core_database.get_db()
    assert next(generator) is fake_session
    with pytest.raises(StopIteration):
        next(generator)
    assert fake_session.closed is True

    class FakeCursor:
        def __init__(self):
            self.commands = []
            self.closed = False

        def execute(self, command):
            self.commands.append(command)

        def close(self):
            self.closed = True

    class FakeConnection:
        def __init__(self):
            self.cursor_obj = FakeCursor()

        def cursor(self):
            return self.cursor_obj

    fake_connection = FakeConnection()
    core_database.set_sqlite_pragma(fake_connection, None)
    assert fake_connection.cursor_obj.commands == ["PRAGMA foreign_keys=ON"]
    assert fake_connection.cursor_obj.closed is True


def test_geocode_location_success_and_failures(monkeypatch):
    class FakeGeolocator:
        def __init__(self):
            self.calls = []

        def geocode(self, query, **kwargs):
            self.calls.append((query, kwargs))
            if query == "123 Test St":
                return SimpleNamespace(latitude=34.0, longitude=-118.0)
            if query == {"postalcode": "90210", "country": "United States"}:
                return SimpleNamespace(latitude=34.0901, longitude=-118.4065)
            return None

    fake_geolocator = FakeGeolocator()
    GeocodingService._geolocator = fake_geolocator
    assert asyncio.run(GeocodingService.geocode_location("123 Test St")) == (34.0, -118.0)
    assert asyncio.run(GeocodingService.geocode_location("90210")) == pytest.approx(
        (34.0901, -118.4065)
    )
    assert fake_geolocator.calls[1][0] == "90210"
    assert fake_geolocator.calls[2][0] == {"postalcode": "90210", "country": "United States"}

    GeocodingService._geolocator = SimpleNamespace(geocode=lambda query, **kwargs: None)
    with pytest.raises(GeocodeError):
        asyncio.run(GeocodingService.geocode_location("missing"))

    for error in (GeocoderTimedOut(), GeocoderUnavailable(), RuntimeError("boom")):
        GeocodingService._geolocator = SimpleNamespace(
            geocode=lambda query, err=error, **kwargs: (_ for _ in ()).throw(err)
        )
        with pytest.raises(GeocodeError):
            asyncio.run(GeocodingService.geocode_location("bad"))


def test_geocode_location_uses_census_fallback(monkeypatch):
    GeocodingService._geolocator = SimpleNamespace(geocode=lambda query, **kwargs: None)

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "result": {
                    "addressMatches": [
                        {
                            "coordinates": {
                                "x": -77.03518753691,
                                "y": 38.89869893252,
                            }
                        }
                    ]
                }
            }

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params):
            assert params["benchmark"] == "Public_AR_Current"
            assert "1600 Pennsylvania Ave" in params["address"]
            return FakeResponse()

    monkeypatch.setattr("app.services.geocoding.httpx.AsyncClient", FakeAsyncClient)

    coords = asyncio.run(
        GeocodingService.geocode_location("1600 Pennsylvania Ave NW, Washington, DC")
    )

    assert coords == pytest.approx((38.89869893252, -77.03518753691))


def test_geocode_locations_updates_rows(db, monkeypatch):
    loc1 = build_location("loc-success", None, None)
    loc2 = build_location("loc-fail", None, None)
    loc2.address = "bad address"
    db.add_all([loc1, loc2])
    db.commit()

    async def fake_geocode(address):
        if address == "bad address":
            raise GeocodeError(address)
        return 35.0, -117.0

    monkeypatch.setattr(GeocodingService, "geocode_location", staticmethod(fake_geocode))

    geocoded = asyncio.run(GeocodingService.geocode_locations(db))
    db.refresh(loc1)
    db.refresh(loc2)

    assert geocoded == 1
    assert loc1.latitude == 35.0
    assert loc2.latitude is None


def test_advanced_weather_merges_sources(monkeypatch):
    service = AdvancedWeatherService()

    async def fake_noaa(latitude, longitude):
        return {
            "source": "NOAA",
            "temperature_f": 80.0,
            "wind_speed_mph": 20.0,
            "wind_direction": "SW",
            "wind_direction_deg": 225,
        }

    async def fake_openmeteo(latitude, longitude):
        return {
            "source": "Open-Meteo",
            "temperature_f": 100.0,
            "humidity_pct": 10.0,
            "wind_speed_mph": 10.0,
            "wind_direction_deg": 180,
            "precipitation_in": 0.1,
            "pressure_mb": 1000.0,
            "dewpoint_f": 40.0,
            "visibility_miles": 8.0,
            "cloud_coverage_pct": 20.0,
        }

    async def fake_meteostat(latitude, longitude):
        return {
            "source": "Meteostat",
            "temperature_f": 90.0,
            "humidity_pct": 30.0,
            "wind_speed_mph": 5.0,
            "wind_direction_deg": 200,
            "pressure_mb": 1010.0,
        }

    monkeypatch.setattr(service, "get_noaa_weather", fake_noaa)
    monkeypatch.setattr(service, "get_openmeteo_weather", fake_openmeteo)
    monkeypatch.setattr(service, "get_meteostat_weather", fake_meteostat)

    merged = asyncio.run(service.merge_weather_sources(34.0, -118.0))
    current = asyncio.run(service.get_current_weather(34.0, -118.0))
    indices = asyncio.run(service.get_fire_weather_indices(34.0, -118.0))

    assert merged.source == "NOAA+Open-Meteo+Meteostat"
    assert merged.temperature_f == pytest.approx(89.5)
    assert merged.humidity_pct == pytest.approx(15.4545454545)
    assert merged.wind_direction_deg == 225
    assert current["source"] == "NOAA+Open-Meteo+Meteostat"
    assert indices["danger_level"] in {"LOW", "MODERATE", "HIGH", "VERY_HIGH", "EXTREME"}


def test_satellite_service_returns_stac_derived_products(monkeypatch):
    service = SatelliteFireService()

    async def fake_search(collections, bbox, start_date, end_date, query=None):
        if collections == ["sentinel-2-l2a"]:
            return {
                "id": "S2-1",
                "collection": "sentinel-2-l2a",
                "properties": {
                    "datetime": "2026-05-21T10:00:00Z",
                    "eo:cloud_cover": 12.0,
                    "s2:nodata_pixel_percentage": 5.0,
                },
                "assets": {
                    "B04": {"href": "B04"},
                    "B08": {"href": "B08"},
                    "B02": {"href": "B02"},
                    "SCL": {"href": "SCL"},
                },
            }

        return {
            "id": "L8-1",
            "collection": "landsat-c2-l2",
            "properties": {
                "datetime": "2026-05-20T10:00:00Z",
                "eo:cloud_cover": 8.0,
            },
            "assets": {"lwir11": {"href": "lwir11"}, "qa_pixel": {"href": "qa_pixel"}},
        }

    async def fake_load(
        collection_id,
        asset,
        latitude,
        longitude,
        reflectance_scale=None,
        scale_hint=None,
        offset_hint=None,
    ):
        href = asset["href"] if asset else None
        if href is None:
            return None
        arrays = {
            "B04": np.ma.array([[0.2, 0.3], [0.25, 0.4]], dtype=float),
            "B08": np.ma.array([[0.6, 0.7], [0.5, 0.8]], dtype=float),
            "B02": np.ma.array([[0.1, 0.15], [0.12, 0.2]], dtype=float),
            "SCL": np.ma.array([[4, 4], [4, 9]], dtype=float),
            "lwir11": np.ma.array([[300.0, 305.0], [310.0, 315.0]], dtype=float),
            "qa_pixel": np.ma.array([[0, 0], [0, 8]], dtype=float),
        }
        return arrays.get(href)

    monkeypatch.setattr(service, "_search_planetary_computer", fake_search)
    monkeypatch.setattr(service, "_load_asset_window", fake_load)

    sentinel = asyncio.run(service.get_sentinel2_ndvi(34.0, -118.0))
    landsat = asyncio.run(service.get_landsat_thermal(34.0, -118.0))

    assert sentinel["scene_id"] == "S2-1"
    assert sentinel["valid_data_pct"] == pytest.approx(95.0)
    assert sentinel["ndvi_mean"] == pytest.approx(0.4111111111)
    assert sentinel["evi_mean"] == pytest.approx(0.40215885, rel=1e-6)
    assert sentinel["confidence"] == pytest.approx(0.66)
    assert landsat["thermal_asset_count"] == 1
    assert landsat["land_surface_temp_k"] == pytest.approx(305.0)
    assert landsat["thermal_hotspot_fraction"] == pytest.approx(0.0)


def test_satellite_service_fetches_authenticated_firms_detections(monkeypatch):
    service = SatelliteFireService()
    service.firms_map_key = "test-map-key"
    service.firms_sources = ("VIIRS_SNPP_NRT",)

    csv_payload = (
        "latitude,longitude,acq_date,acq_time,confidence,frp,instrument\n"
        "34.1,-118.2,2026-05-21,0930,h,12.5,VIIRS\n"
    )
    captured = {}

    class FakeResponse:
        text = csv_payload

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, timeout):
            captured["url"] = url
            return FakeResponse()

    monkeypatch.setattr("app.services.satellite_service.httpx.AsyncClient", FakeAsyncClient)

    detections = asyncio.run(service.get_firms_detections(34.0, -118.0, radius_km=25, days=7))

    assert "/test-map-key/VIIRS_SNPP_NRT/" in captured["url"]
    assert captured["url"].endswith("/5")
    assert len(detections) == 1
    assert detections[0].sensor == "VIIRS"
    assert detections[0].confidence == pytest.approx(85.0)
    assert detections[0].radiative_power == pytest.approx(12.5)


def test_comprehensive_data_uses_v1_models_and_avoids_dummy_defaults(monkeypatch):
    service = ComprehensiveDataService()
    assert service.spread_predictor.__class__.__name__ == "FireSpreadPredictor"
    assert service.anomaly_detector.__class__.__name__ == "AnomalyDetector"
    assert service.risk_scorer.__class__.__name__ == "RiskScorer"

    async def fake_merge(latitude, longitude):
        return WeatherData(
            timestamp=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
            temperature_f=88.0,
            humidity_pct=18.0,
            wind_speed_mph=22.0,
            wind_direction_deg=225,
            precipitation_in=0.0,
            pressure_mb=1009.0,
            dewpoint_f=45.0,
            visibility_miles=9.0,
            cloud_coverage_pct=10,
            source="NOAA",
        )

    async def fake_sentinel(latitude, longitude, start_date=None, end_date=None):
        return {"source": "Sentinel-2", "scene_id": "S2-1", "cloud_coverage": 8.0}

    async def fake_landsat(latitude, longitude):
        return {
            "source": "Landsat",
            "scene_id": "L8-1",
            "cloud_coverage": 15.0,
            "thermal_asset_count": 2,
        }

    async def fake_topography(latitude, longitude):
        return {"elevation_m": 123.0, "slope_deg": None, "aspect_deg": None, "source": "USGS_3DEP"}

    async def fake_empty(*args, **kwargs):
        return {}

    monkeypatch.setattr(service.weather_service, "merge_weather_sources", fake_merge)
    monkeypatch.setattr(
        service.weather_service,
        "calculate_fire_weather_index_v2",
        lambda weather: {
            "fwi": 14.0,
            "danger_level": "MODERATE",
        },
    )
    monkeypatch.setattr(service.satellite_service, "get_sentinel2_ndvi", fake_sentinel)
    monkeypatch.setattr(service.satellite_service, "get_landsat_thermal", fake_landsat)
    monkeypatch.setattr(service.topo_service, "get_elevation_and_slope", fake_topography)
    monkeypatch.setattr(service.topo_service, "get_fuel_type", fake_empty)
    monkeypatch.setattr(service.veg_service, "get_ndvi", fake_empty)
    monkeypatch.setattr(service.veg_service, "get_moisture_stress", fake_empty)
    monkeypatch.setattr(service.spread_predictor, "predict_spread_rate", lambda fire, history: 1.75)
    monkeypatch.setattr(
        service.anomaly_detector,
        "detect_anomaly",
        lambda fire, history: {"is_anomaly": True, "anomaly_score": -0.2},
    )
    monkeypatch.setattr(
        service.risk_scorer,
        "calculate_combined_threat_score",
        lambda location, fire, distance, spread_rate=None: 67.0,
    )
    monkeypatch.setattr(service.risk_scorer, "get_threat_description", lambda score: "High")

    data = asyncio.run(
        service.get_comprehensive_fire_data(
            fire_id="comp-fire",
            fire_lat=34.1,
            fire_lon=-118.2,
            location_lat=34.0,
            location_lon=-118.0,
            fire_data={
                "acres_burned": 100.0,
                "percent_contained": 20.0,
                "spread_rate_mph": 1.2,
                "history": [
                    {
                        "timestamp": "2026-05-21T10:00:00+00:00",
                        "distance_miles": 8.0,
                        "bearing_degrees": 90.0,
                    },
                    {
                        "timestamp": "2026-05-21T11:00:00+00:00",
                        "distance_miles": 6.0,
                        "bearing_degrees": 110.0,
                    },
                ],
            },
            location_data={"location_id": "watch-home"},
        )
    )

    assert data.weather_source == "NOAA"
    assert data.predicted_spread_rate_mph == pytest.approx(1.75)
    assert data.threat_score == pytest.approx(67.0)
    assert data.ndvi is None
    assert data.lst_kelvin is None
    assert data.fuel_model is None
    assert 0.9 <= data.satellite_confidence <= 1.0
    assert data.data_completeness < 100.0


def test_comprehensive_data_consumes_external_intelligence(monkeypatch):
    service = ComprehensiveDataService()

    async def fake_merge(latitude, longitude):
        return WeatherData(
            timestamp=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
            temperature_f=84.0,
            humidity_pct=22.0,
            wind_speed_mph=18.0,
            wind_direction_deg=180,
            precipitation_in=0.0,
            pressure_mb=1010.0,
            dewpoint_f=42.0,
            visibility_miles=10.0,
            cloud_coverage_pct=5,
            source="NOAA",
        )

    async def fake_empty_dict(*args, **kwargs):
        return {}

    async def fake_empty_list(*args, **kwargs):
        return []

    async def fake_intelligence(fire_lat, fire_lon, location_lat, location_lon):
        return {
            "incident_perimeters": {
                "source": "Configured perimeter feed",
                "incident_count": 1,
                "nearest_incident": {
                    "name": "Bridge Fire",
                    "acres": 400.0,
                    "percent_contained": 15.0,
                    "growth_acres_24h": 60.0,
                    "distance_miles": 0.6,
                },
            },
            "road_closures": {
                "source": "Configured road closure feed",
                "closure_count_nearby": 1,
                "nearest_closure": {"distance_miles": 2.5, "road_name": "CA-2", "status": "CLOSED"},
            },
            "lightning_nowcast": {
                "source": "Configured lightning nowcast feed",
                "strikes_nearby_count": 4,
                "nearest_strike": {"distance_miles": 9.0},
                "elevated_ignition_risk": True,
            },
            "feed_status": {
                "incident_perimeters": "ok",
                "road_closures": "ok",
                "lightning_nowcast": "ok",
            },
            "reliability": {"status": "ok", "success_ratio": 1.0},
        }

    monkeypatch.setattr(service.weather_service, "merge_weather_sources", fake_merge)
    monkeypatch.setattr(
        service.weather_service, "calculate_fire_weather_index_v2", lambda weather: {"fwi": 11.0}
    )
    monkeypatch.setattr(service.satellite_service, "get_sentinel2_ndvi", fake_empty_dict)
    monkeypatch.setattr(service.satellite_service, "get_landsat_thermal", fake_empty_dict)
    monkeypatch.setattr(service.satellite_service, "get_firms_detections", fake_empty_list)
    monkeypatch.setattr(service.topo_service, "get_elevation_and_slope", fake_empty_dict)
    monkeypatch.setattr(service.topo_service, "get_fuel_type", fake_empty_dict)
    monkeypatch.setattr(service.veg_service, "get_ndvi", fake_empty_dict)
    monkeypatch.setattr(service.veg_service, "get_moisture_stress", fake_empty_dict)
    monkeypatch.setattr(
        service.external_intelligence_service,
        "collect_for_fire_and_location",
        fake_intelligence,
    )
    monkeypatch.setattr(service.spread_predictor, "predict_spread_rate", lambda fire, history: 1.0)
    monkeypatch.setattr(
        service.anomaly_detector,
        "detect_anomaly",
        lambda fire, history: {"is_anomaly": False, "anomaly_score": -0.1},
    )
    monkeypatch.setattr(
        service.risk_scorer,
        "calculate_combined_threat_score",
        lambda location, fire, distance, spread_rate=None: 45.0,
    )
    monkeypatch.setattr(service.risk_scorer, "get_threat_description", lambda score: "Moderate")

    data = asyncio.run(
        service.get_comprehensive_fire_data(
            fire_id="comp-intel-fire",
            fire_lat=34.1,
            fire_lon=-118.2,
            location_lat=34.0,
            location_lon=-118.0,
            fire_data={
                "history": [
                    {
                        "timestamp": "2026-05-21T10:00:00+00:00",
                        "distance_miles": 8.0,
                        "bearing_degrees": 90.0,
                    },
                    {
                        "timestamp": "2026-05-21T11:00:00+00:00",
                        "distance_miles": 7.0,
                        "bearing_degrees": 100.0,
                    },
                ]
            },
            location_data={"location_id": "watch-home"},
        )
    )

    assert data.acres_burned == pytest.approx(400.0)
    assert data.percent_contained == pytest.approx(15.0)
    assert data.road_closure_count_nearby == 1
    assert data.lightning_strikes_nearby == 4
    assert data.elevated_lightning_risk is True
    assert data.external_reliability_status == "ok"
    assert (
        data.external_intelligence["incident_perimeters"]["nearest_incident"]["name"]
        == "Bridge Fire"
    )


def test_comprehensive_data_satellite_gathering_includes_firms(monkeypatch):
    service = ComprehensiveDataService()

    async def fake_sentinel(latitude, longitude, start_date=None, end_date=None):
        return {
            "source": "Sentinel-2",
            "scene_id": "S2-1",
            "cloud_coverage": 40.0,
            "ndvi_mean": 0.44,
        }

    async def fake_landsat(latitude, longitude):
        return {
            "source": "Landsat",
            "scene_id": "L8-1",
            "cloud_coverage": 50.0,
            "land_surface_temp_k": 305.0,
        }

    async def fake_firms(latitude, longitude, radius_km=50, days=7):
        return [
            FireDetection(
                latitude=latitude,
                longitude=longitude,
                confidence=88.0,
                sensor="VIIRS",
                timestamp=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
                radiative_power=18.0,
            )
        ]

    monkeypatch.setattr(service.satellite_service, "get_sentinel2_ndvi", fake_sentinel)
    monkeypatch.setattr(service.satellite_service, "get_landsat_thermal", fake_landsat)
    monkeypatch.setattr(service.satellite_service, "get_firms_detections", fake_firms)

    satellite = asyncio.run(service._gather_satellite_data(34.0, -118.0))

    assert satellite["ndvi"] == pytest.approx(0.44)
    assert satellite["lst_kelvin"] == pytest.approx(305.0)
    assert satellite["fire_detection_count"] == 1
    assert satellite["mean_radiative_power_mw"] == pytest.approx(18.0)
    assert satellite["confidence"] == pytest.approx(0.88)
    assert satellite["source"] == "Sentinel-2+Landsat+FIRMS:VIIRS"


def test_comprehensive_data_source_inventory_reports_configured_optional_feeds(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(
        settings,
        "FIRE_PERIMETER_GEOJSON_URL",
        "https://example.com/perimeters.geojson",
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "PURPLEAIR_API_KEY",
        "purpleair-key",
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "LIGHTNING_NOWCAST_URL",
        "https://example.com/lightning.json",
        raising=False,
    )

    inventory = ComprehensiveDataService.build_data_source_inventory()

    operational = {
        item["name"]: item for item in inventory["implemented_now"]["operational_overlays"]
    }
    weather_air_quality = {
        item["name"]: item for item in inventory["implemented_now"]["weather_and_air_quality"]
    }

    assert operational["Configured incident perimeter GeoJSON"]["status"] == "configured"
    assert operational["Configured lightning nowcast feed"]["status"] == "configured"
    assert weather_air_quality["PurpleAir neighborhood sensors"]["status"] == "configured"
    assert any(
        gap["name"] == "LANDFIRE fuels, canopy, and fuel loading"
        for gap in inventory["highest_impact_gaps"]
    )


def test_fire_data_fetch_process_and_movement(db, monkeypatch):
    async def no_firms(cls, locations):
        return []

    monkeypatch.setattr(FireDataService, "_fetch_firms_near_locations", classmethod(no_firms))

    valid_payload = [{"Name": "Alpha", "UniqueId": "fire-1", "Latitude": 34.1, "Longitude": -118.1}]
    monkeypatch.setattr(
        "app.services.fire_data.httpx.AsyncClient",
        lambda: FakeAsyncClient(payload=valid_payload),
    )
    assert asyncio.run(FireDataService.fetch_fires_from_api())[0]["UniqueId"] == "fire-1"

    monkeypatch.setattr(
        "app.services.fire_data.httpx.AsyncClient",
        lambda: FakeAsyncClient(payload={"not": "a list"}),
    )
    with pytest.raises(FireDataFetchError):
        asyncio.run(FireDataService.fetch_fires_from_api())

    monkeypatch.setattr(
        "app.services.fire_data.httpx.AsyncClient",
        lambda: FakeAsyncClient(error=httpx.HTTPError("boom")),
    )
    with pytest.raises(FireDataFetchError):
        asyncio.run(FireDataService.fetch_fires_from_api())

    location = build_location("watch-home", 34.0, -118.0)
    db.add(location)
    db.commit()

    async def fake_fetch():
        return [
            {
                "Name": "Near Fire",
                "UniqueId": "near-1",
                "Latitude": 34.1,
                "Longitude": -118.1,
                "AcresBurned": 1200,
                "PercentContained": 5,
            },
            {
                "Name": "Near Fire",
                "UniqueId": "near-1",
                "Latitude": 34.1,
                "Longitude": -118.1,
            },
            {
                "Name": "Invalid Fire",
                "UniqueId": "bad-1",
                "Latitude": 0,
                "Longitude": 0,
            },
        ]

    original_calculate_movement = FireDataService.calculate_movement

    monkeypatch.setattr(FireDataService, "fetch_fires_from_api", staticmethod(fake_fetch))
    monkeypatch.setattr(
        FireDataService,
        "calculate_movement",
        classmethod(lambda cls, fire_id, location_id, current_distance: (True, 3.4)),
    )

    processed = asyncio.run(FireDataService.process_fire_data(db, [location]))
    assert len(processed) == 1
    assert processed[0]["fire_id"] == "near-1"
    assert processed[0]["movement_data"]["watch-home"]["moving_towards"] is True

    async def broken_fetch():
        raise ValueError("broken")

    monkeypatch.setattr(FireDataService, "fetch_fires_from_api", staticmethod(broken_fetch))
    with pytest.raises(FireDataFetchError):
        asyncio.run(FireDataService.process_fire_data(db, [location]))

    monkeypatch.setattr(FireDataService, "calculate_movement", original_calculate_movement)

    FireDataService._distance_history_cache.clear()
    moving_towards, speed = FireDataService.calculate_movement("fire-x", "loc-x", 10.0)
    assert moving_towards is False and speed == 0.0
    FireDataService._distance_history_cache["fire-x_loc-x"][0]["timestamp"] = datetime.now(
        UTC
    ) - timedelta(hours=1)
    moving_towards, speed = FireDataService.calculate_movement("fire-x", "loc-x", 8.0)
    assert moving_towards is True
    assert speed > 0


def test_fire_data_process_merges_configured_perimeter_metadata(db, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(
        settings,
        "FIRE_PERIMETER_GEOJSON_URL",
        "https://example.com/perimeters.geojson",
        raising=False,
    )

    async def no_firms(cls, locations):
        return []

    async def fake_fetch():
        return [
            {
                "Name": "Bridge Fire",
                "UniqueId": "calfire-bridge",
                "Latitude": 34.15,
                "Longitude": -118.15,
                "AcresBurned": 120.0,
                "PercentContained": 5.0,
                "Source": "CAL_FIRE",
            }
        ]

    geojson_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-118.16, 34.14],
                            [-118.14, 34.14],
                            [-118.14, 34.16],
                            [-118.16, 34.16],
                            [-118.16, 34.14],
                        ]
                    ],
                },
                "properties": {
                    "IncidentName": "Bridge Fire",
                    "IRWINID": "irwin-bridge",
                    "DailyAcres": 950.0,
                    "PercentContained": 12.0,
                    "DailyGrowth": 75.0,
                    "DateCurrent": "2026-05-21T12:00:00Z",
                },
            }
        ],
    }

    monkeypatch.setattr(FireDataService, "fetch_fires_from_api", staticmethod(fake_fetch))
    monkeypatch.setattr(FireDataService, "_fetch_firms_near_locations", classmethod(no_firms))
    monkeypatch.setattr(
        "app.services.external_intelligence.httpx.AsyncClient",
        lambda: FakeAsyncClient(payload=geojson_payload),
    )
    monkeypatch.setattr(
        FireDataService,
        "calculate_movement",
        classmethod(lambda cls, fire_id, location_id, current_distance: (False, 0.0)),
    )

    location = build_location("perimeter-home", 34.05, -118.05)
    db.add(location)
    db.commit()

    processed = asyncio.run(FireDataService.process_fire_data(db, [location]))

    assert len(processed) == 1
    fire = processed[0]
    assert fire["acres_burned"] == pytest.approx(950.0)
    assert fire["percent_contained"] == pytest.approx(12.0)
    assert fire["polygon_data"]["type"] == "Polygon"
    assert fire["extra_metadata"]["intelligence_source"] == "Configured perimeter feed"
    assert fire["extra_metadata"]["growth_acres_24h"] == pytest.approx(75.0)
    assert "PERIMETER" in fire["extra_metadata"]["merged_sources"]


def test_fire_data_process_uses_firms_when_primary_feeds_are_empty(db, monkeypatch):
    location = build_location("watch-home", 34.0, -118.0)
    db.add(location)
    db.commit()

    async def fake_fetch():
        return []

    async def fake_firms(cls, locations):
        return [
            {
                "Name": "VIIRS satellite hotspot",
                "UniqueId": "firms-viirs-hotspot",
                "Latitude": 34.05,
                "Longitude": -118.05,
                "AcresBurned": 0.0,
                "PercentContained": 0.0,
                "Source": "FIRMS:VIIRS",
            }
        ]

    monkeypatch.setattr(FireDataService, "fetch_fires_from_api", staticmethod(fake_fetch))
    monkeypatch.setattr(FireDataService, "_fetch_firms_near_locations", classmethod(fake_firms))

    processed = asyncio.run(FireDataService.process_fire_data(db, [location]))

    assert len(processed) == 1
    assert processed[0]["fire_id"] == "firms-viirs-hotspot"
    assert processed[0]["name"] == "VIIRS satellite hotspot"


def test_fire_data_process_merges_nearby_firms_into_named_incident(db, monkeypatch):
    location = build_location("watch-home", 34.0, -118.0)
    db.add(location)
    db.commit()

    async def fake_fetch():
        return [
            {
                "Name": "Merge Fire",
                "UniqueId": "cal-merge-1",
                "Latitude": 34.101,
                "Longitude": -118.201,
                "AcresBurned": 500.0,
                "PercentContained": 20.0,
                "Source": "CAL_FIRE",
            }
        ]

    async def fake_firms(cls, locations):
        return [
            {
                "Name": "VIIRS satellite hotspot",
                "UniqueId": "firms-merge-hotspot",
                "Latitude": 34.102,
                "Longitude": -118.202,
                "AcresBurned": 0.0,
                "PercentContained": 0.0,
                "Source": "FIRMS:VIIRS",
                "ObservedAt": "2026-05-21T10:00:00+00:00",
                "Confidence": 92.0,
            }
        ]

    monkeypatch.setattr(FireDataService, "fetch_fires_from_api", staticmethod(fake_fetch))
    monkeypatch.setattr(FireDataService, "_fetch_firms_near_locations", classmethod(fake_firms))

    processed = asyncio.run(FireDataService.process_fire_data(db, [location]))

    assert len(processed) == 1
    assert processed[0]["fire_id"] == "cal-merge-1"
    assert processed[0]["name"] == "Merge Fire"


def test_fire_data_fetch_multisource_merge(monkeypatch):
    async def fake_calfire(client):
        return [
            {
                "Name": "Merge Fire",
                "UniqueId": "cal-merge-1",
                "Latitude": 34.101,
                "Longitude": -118.201,
                "AcresBurned": 500.0,
                "PercentContained": 20.0,
                "Source": "CAL_FIRE",
            }
        ]

    async def fake_eonet(client):
        return [
            {
                "Name": "Merge Fire",
                "UniqueId": "EONET_1",
                "Latitude": 34.104,
                "Longitude": -118.204,
                "AcresBurned": 0.0,
                "PercentContained": 0.0,
                "Source": "EONET",
            }
        ]

    monkeypatch.setattr(FireDataService, "_fetch_calfire_fires", staticmethod(fake_calfire))
    monkeypatch.setattr(
        FireDataService,
        "_fetch_eonet_fires",
        classmethod(lambda cls, client: fake_eonet(client)),
    )

    merged = asyncio.run(FireDataService.fetch_fires_from_api())

    assert len(merged) == 1
    assert merged[0]["UniqueId"] == "cal-merge-1"
    assert merged[0]["Source"] == "CAL_FIRE+EONET"
    assert merged[0]["AcresBurned"] == pytest.approx(500.0)


def test_fire_prediction_and_anomaly_training_paths():
    predictor = FireSpreadPredictor()
    detector = AnomalyDetector()
    fire = build_fire(acres=1000, contained=40)
    history = [
        SimpleNamespace(
            timestamp=datetime.now(UTC) - timedelta(hours=2),
            distance_miles=8.0,
            bearing_degrees=45.0,
        ),
        SimpleNamespace(
            timestamp=datetime.now(UTC) - timedelta(hours=1),
            distance_miles=6.0,
            bearing_degrees=60.0,
        ),
        SimpleNamespace(
            timestamp=datetime.now(UTC),
            distance_miles=4.0,
            bearing_degrees=90.0,
        ),
    ]

    features = predictor.extract_fire_features(fire, history)
    assert features["num_updates"] == 3

    predictor.train(
        [
            {"features": features, "spread_rate": 1.5},
            {"features": {**features, "current_distance": 3.0}, "spread_rate": 2.0},
        ]
    )
    assert predictor.is_trained is True
    assert predictor.predict_spread_rate(fire, history) >= 0.0
    assert predictor.predict_containment_time(fire, history)["hours_until_contained"] >= 0.0

    anomaly_features = detector.extract_fire_behavior_features(fire, history)
    detector.train(
        [
            {"features": anomaly_features},
            {
                "features": {
                    **anomaly_features,
                    "mean_spread_rate": anomaly_features["mean_spread_rate"] + 1,
                }
            },
        ]
    )
    assert detector.is_trained is True
    result = detector.detect_anomaly(fire, history)
    assert "features" in result


def test_ml_analysis_service_paths(monkeypatch):
    service = MLFireAnalysisService()
    fire = build_fire()
    location = build_location()
    dummy_db = SimpleNamespace(query=lambda model: [])

    async def fake_current_weather(latitude, longitude):
        return {"temperature_f": 95, "humidity_pct": 15, "wind_speed_mph": 25}

    async def fake_fire_weather_indices(latitude, longitude):
        return {"fwi_index": 70, "danger_level": "VERY_HIGH"}

    async def fake_forecast(latitude, longitude, days=7):
        return {
            "forecast": [
                {"date": "2026-05-21", "temp_max_f": 90, "humidity_pct": 20, "wind_speed_mph": 10},
                {"date": "2026-05-22", "temp_max_f": 70, "humidity_pct": 60, "wind_speed_mph": 5},
            ]
        }

    monkeypatch.setattr(service.weather_service, "get_current_weather", fake_current_weather)
    monkeypatch.setattr(
        service.weather_service, "get_fire_weather_indices", fake_fire_weather_indices
    )
    monkeypatch.setattr(service.weather_service, "get_forecast", fake_forecast)
    monkeypatch.setattr(service.predictor, "predict_spread_rate", lambda fire, history: 2.0)
    monkeypatch.setattr(
        service.predictor,
        "predict_impact_timeline",
        lambda fire, history, target_distance: {"hours_until_impact": 5},
    )
    monkeypatch.setattr(
        service.predictor,
        "predict_containment_time",
        lambda fire, history: {"hours_until_contained": 12},
    )
    monkeypatch.setattr(
        service.predictor, "predict_direction_probabilities", lambda fire, history: {0: 1.0}
    )
    monkeypatch.setattr(
        service.anomaly_detector,
        "detect_anomaly",
        lambda fire, history: {"is_anomaly": False, "reasons": []},
    )
    monkeypatch.setattr(service.anomaly_detector, "get_anomaly_insight", lambda fire, history: None)

    analysis = asyncio.run(service.analyze_fire_threat(fire, location, 5.0, dummy_db))
    assert analysis["threat_score"] >= 0
    assert analysis["weather"]["fire_danger_level"] == "VERY_HIGH"

    async def fake_analyze(fire, location, distance, db):
        return {"threat_score": 20, "predictions": {"adjusted_spread_rate_mph": 1.0}}

    monkeypatch.setattr(service, "analyze_fire_threat", fake_analyze)
    monitor_result = asyncio.run(service.get_evacuation_optimization(location, [fire], dummy_db))
    assert monitor_result["urgency_level"] == "MONITOR"

    threat_scores = [(20, "MONITOR"), (40, "PREPARE"), (60, "WARNING"), (80, "EVACUATE")]
    for score, expected in threat_scores:

        async def score_analyze(fire, location, distance, db, threat_score=score):
            return {"threat_score": threat_score, "predictions": {"adjusted_spread_rate_mph": 2.5}}

        monkeypatch.setattr(service, "analyze_fire_threat", score_analyze)
        result = asyncio.run(service.get_evacuation_optimization(location, [fire], dummy_db))
        assert result["urgency_level"] == expected

    monkeypatch.setattr(service.weather_service, "get_forecast", fake_forecast)
    predictive = asyncio.run(service.get_predictive_analytics(location, forecast_hours=48))
    assert len(predictive["predictions"]) == 2

    async def no_forecast(latitude, longitude, days=7):
        return None

    monkeypatch.setattr(service.weather_service, "get_forecast", no_forecast)
    assert asyncio.run(service.get_predictive_analytics(location, forecast_hours=24))["error"]


def test_ml_analysis_service_applies_external_intelligence(monkeypatch):
    service = MLFireAnalysisService()
    fire = build_fire(acres=120.0, contained=5.0)
    location = build_location()
    dummy_db = SimpleNamespace(query=lambda model: [])
    captured: dict[str, float] = {}

    async def fake_current_weather(latitude, longitude):
        return {"temperature_f": 92, "humidity_pct": 16, "wind_speed_mph": 20}

    async def fake_fire_weather_indices(latitude, longitude):
        return {"fwi_index": 55, "danger_level": "HIGH"}

    async def fake_intelligence(fire_lat, fire_lon, location_lat, location_lon):
        return {
            "incident_perimeters": {
                "source": "Configured perimeter feed",
                "incident_count": 1,
                "nearest_incident": {
                    "name": "Bridge Fire",
                    "acres": 2500.0,
                    "percent_contained": 8.0,
                    "growth_acres_24h": 200.0,
                    "distance_miles": 0.4,
                },
            },
            "road_closures": {
                "source": "Configured road closure feed",
                "closure_count_nearby": 2,
                "nearest_closure": {
                    "distance_miles": 3.0,
                    "road_name": "Angeles Crest",
                    "status": "CLOSED",
                },
            },
            "lightning_nowcast": {
                "source": "Configured lightning nowcast feed",
                "strikes_nearby_count": 3,
                "nearest_strike": {"distance_miles": 12.0},
                "elevated_ignition_risk": True,
            },
            "feed_status": {
                "incident_perimeters": "ok",
                "road_closures": "ok",
                "lightning_nowcast": "ok",
            },
            "reliability": {"status": "ok", "success_ratio": 1.0},
        }

    def fake_fire_risk(fire_obj):
        captured["acres_burned"] = float(fire_obj.acres_burned)
        captured["percent_contained"] = float(fire_obj.percent_contained)
        return 40.0

    monkeypatch.setattr(service.weather_service, "get_current_weather", fake_current_weather)
    monkeypatch.setattr(
        service.weather_service, "get_fire_weather_indices", fake_fire_weather_indices
    )
    monkeypatch.setattr(
        service.external_intelligence_service,
        "collect_for_fire_and_location",
        fake_intelligence,
    )
    monkeypatch.setattr(service.predictor, "predict_spread_rate", lambda fire_obj, history: 2.0)
    monkeypatch.setattr(
        service.predictor,
        "predict_impact_timeline",
        lambda fire_obj, history, target_distance: {"hours_until_impact": 5},
    )
    monkeypatch.setattr(
        service.predictor,
        "predict_containment_time",
        lambda fire_obj, history: {"hours_until_contained": 12},
    )
    monkeypatch.setattr(
        service.predictor, "predict_direction_probabilities", lambda fire_obj, history: {0: 1.0}
    )
    monkeypatch.setattr(
        service.anomaly_detector,
        "detect_anomaly",
        lambda fire_obj, history: {"is_anomaly": False, "reasons": []},
    )
    monkeypatch.setattr(
        service.anomaly_detector, "get_anomaly_insight", lambda fire_obj, history: None
    )
    monkeypatch.setattr(service.risk_scorer, "calculate_fire_risk_score", fake_fire_risk)
    monkeypatch.setattr(
        service.risk_scorer,
        "calculate_location_vulnerability_score",
        lambda location_obj, fire_obj, distance: 50.0,
    )
    monkeypatch.setattr(
        service.risk_scorer,
        "calculate_combined_threat_score",
        lambda location_obj, fire_obj, distance, spread_rate=None: 60.0,
    )
    monkeypatch.setattr(service.risk_scorer, "get_threat_description", lambda score: "High")
    monkeypatch.setattr(
        service.risk_scorer, "get_recommendations", lambda score: ["Baseline recommendation"]
    )

    analysis = asyncio.run(service.analyze_fire_threat(fire, location, 5.0, dummy_db))

    assert captured["acres_burned"] == pytest.approx(2500.0)
    assert captured["percent_contained"] == pytest.approx(8.0)
    assert analysis["threat_score"] > 60.0
    assert analysis["external_intelligence"]["road_closures"]["closure_count_nearby"] == 2
    assert any(
        "Road closures reported nearby" in item for item in analysis["external_risk_modifiers"]
    )
    assert any("alternate evacuation routes" in item for item in analysis["recommendations"])
    assert analysis["predictions"]["external_risk_score_boost"] > 0.0


def test_ml_analysis_service_uses_historical_benchmark(db, monkeypatch):
    service = MLFireAnalysisService()
    location = build_location("bench-home", 34.0, -118.0)
    fire = build_fire("bench-fire", acres=1200, contained=15, latitude=34.1, longitude=-118.1)
    db.add_all([location, fire])
    db.commit()
    db.refresh(location)
    db.refresh(fire)

    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    distances = [10.0, 8.7, 7.5, 6.4, 5.5, 4.7, 4.0, 3.4]
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
            )
        )
    db.commit()

    async def fake_current_weather(latitude, longitude):
        return {"temperature_f": 82, "humidity_pct": 20, "wind_speed_mph": 12}

    async def fake_fire_weather_indices(latitude, longitude):
        return {"fwi_index": 30, "danger_level": "HIGH"}

    monkeypatch.setattr(service.weather_service, "get_current_weather", fake_current_weather)
    monkeypatch.setattr(
        service.weather_service, "get_fire_weather_indices", fake_fire_weather_indices
    )
    monkeypatch.setattr(
        service.anomaly_detector,
        "detect_anomaly",
        lambda fire_obj, history: {"is_anomaly": False, "reasons": []},
    )
    monkeypatch.setattr(
        service.anomaly_detector, "get_anomaly_insight", lambda fire_obj, history: None
    )

    analysis = asyncio.run(service.analyze_fire_threat(fire, location, 3.4, db))

    assert service.predictor.is_trained is True
    assert analysis["model_selection"]["benchmark_ran"] is True
    assert analysis["model_selection"]["training_samples"] >= 6
    assert analysis["predictions"]["spread_model"] == service.predictor.spread_model_name


def test_smoke_service_parses_hms_kml_and_scores_location():
    service = SmokeDataService()
    kml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
        <kml xmlns=\"http://www.opengis.net/kml/2.2\">
            <Document>
                <Placemark>
                    <description><![CDATA[<div>Start Time: 2026140 1200UTC<br>End Time: 2026140 1500UTC<br>Density: Heavy<br>Satellite: GOES-WEST</div>]]></description>
                    <styleUrl>#Smoke_Heavy_style</styleUrl>
                    <Polygon>
                        <outerBoundaryIs>
                            <LinearRing>
                                <coordinates>
                                    -118.20,34.00,0 -117.80,34.00,0 -117.80,34.20,0 -118.20,34.20,0 -118.20,34.00,0
                                </coordinates>
                            </LinearRing>
                        </outerBoundaryIs>
                    </Polygon>
                </Placemark>
            </Document>
        </kml>"""

    plumes = service._parse_kml(kml)
    assert len(plumes) == 1
    assert plumes[0].density == "HEAVY"

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(service, "_load_plumes_for_date", lambda day: plumes)
    assessment = service.get_smoke_assessment(34.1, -118.0, lookback_days=1)
    monkeypatch.undo()

    assert assessment is not None
    assert assessment["smoke_density"] == "Heavy"
    assert assessment["smoke_scope"] == "overhead"


def test_air_quality_service_parses_openmeteo_response(monkeypatch):
    service = AirQualityService()

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "utc_offset_seconds": -25200,
                "current": {
                    "time": "2026-05-21T03:00",
                    "us_aqi": 83,
                    "pm2_5": 46.6,
                },
            }

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params):
            assert params["current"] == "us_aqi,pm2_5"
            return FakeResponse()

    monkeypatch.setattr("app.services.air_quality.httpx.Client", FakeClient)

    assessment = service.get_air_quality_assessment(34.05, -118.25)

    assert assessment is not None
    assert assessment["air_quality_index"] == 83
    assert assessment["pm25_ug_m3"] == pytest.approx(46.6)
    assert assessment["air_quality_category"] == "Moderate"


def test_air_quality_service_merges_airnow_and_openmeteo(monkeypatch):
    service = AirQualityService()
    service.airnow_api_key = "test-key"

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params):
            if "open-meteo" in url:
                return FakeResponse(
                    {
                        "utc_offset_seconds": 0,
                        "current": {
                            "time": "2026-05-21T12:00",
                            "us_aqi": 90,
                            "pm2_5": 32.0,
                        },
                    }
                )

            assert params["API_KEY"] == "test-key"
            return FakeResponse(
                [
                    {
                        "DateObserved": "2026-05-21",
                        "HourObserved": 12,
                        "ParameterName": "O3",
                        "AQI": 140,
                        "Category": {"Name": "Unhealthy for Sensitive Groups"},
                    },
                    {
                        "DateObserved": "2026-05-21",
                        "HourObserved": 12,
                        "ParameterName": "PM2.5",
                        "AQI": 125,
                        "Concentration": 41.0,
                    },
                ]
            )

    monkeypatch.setattr("app.services.air_quality.httpx.Client", FakeClient)

    assessment = service.get_air_quality_assessment(34.05, -118.25)

    assert assessment is not None
    assert float(assessment["air_quality_index"]) >= 115
    assert float(assessment["pm25_ug_m3"]) > 35
    assert "EPA AirNow" in str(assessment["air_quality_source"])
    assert float(assessment["air_quality_confidence"]) > 0.7


def test_main_helpers_and_handlers(client):
    assert client.get("/").status_code == 200
    app_shell_response = client.get("/app")
    assert app_shell_response.status_code == 200
    assert "localStorage" in app_shell_response.text
    invalid_response = client.post(
        "/api/v1/locations", json={"location_id": "broken", "address": "123"}
    )
    assert invalid_response.status_code == 422
    assert root()["ui"] == "/app"
    assert root()["docs"] == "/docs"

    request = Request({"type": "http", "method": "GET", "path": "/boom", "headers": []})
    firewatch_response = asyncio.run(
        firewatch_exception_handler(request, FirewatchException("boom", code="TEST"))
    )
    assert firewatch_response.status_code == 400

    validation_response = asyncio.run(
        validation_exception_handler(
            request,
            RequestValidationError(
                [{"loc": ("body", "field"), "msg": "bad", "type": "value_error"}]
            ),
        )
    )
    assert validation_response.status_code == 422


def test_fetch_fires_periodically_and_lifespan(db, monkeypatch):
    class StopLoop(BaseException):
        pass

    location = build_location("monitored-home", 34.0, -118.0)
    db.add(location)
    db.commit()

    async def fake_process_fire_data(db_session, locations):
        return [
            {
                "fire_id": "loop-fire",
                "name": "Loop Fire",
                "latitude": 34.2,
                "longitude": -118.2,
                "acres_burned": 6000,
                "percent_contained": 5,
                "distances": {"monitored-home": 1.0},
                "bearings": {"monitored-home": 90.0},
                "directions": {"monitored-home": "E (90°)"},
                "movement_data": {"monitored-home": {"moving_towards": True, "speed": 4.0}},
            }
        ]

    async def stop_sleep(seconds):
        raise StopLoop()

    monkeypatch.setattr("app.main.asyncio.sleep", stop_sleep)
    monkeypatch.setattr(
        "app.main.FireDataService.process_fire_data", staticmethod(fake_process_fire_data)
    )

    with pytest.raises(StopLoop):
        asyncio.run(fetch_fires_periodically(db))

    assert db.query(Fire).filter(Fire.fire_id == "loop-fire").first() is not None
    assert db.query(DistanceHistory).count() == 1
    assert db.query(Alert).count() == 1

    async def broken_process_fire_data(db_session, locations):
        raise RuntimeError("broken")

    monkeypatch.setattr(
        "app.main.FireDataService.process_fire_data", staticmethod(broken_process_fire_data)
    )
    with pytest.raises(StopLoop):
        asyncio.run(fetch_fires_periodically(db))

    class FakeQuery:
        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return []  # pragma: no cover

        def first(self):
            return None

    class FakeSession:
        def __init__(self):
            self.closed = False
            self.added = []

        def query(self, model):
            return FakeQuery()

        def add(self, obj):
            self.added.append(obj)

        def commit(self):
            return None

        def close(self):
            self.closed = True

    fake_sessions = []

    def fake_session_local():
        session = FakeSession()
        fake_sessions.append(session)
        return session

    class FakeTask:
        def __init__(self, coro):
            self.coro = coro
            self.cancelled = False
            coro.close()

        def cancel(self):
            self.cancelled = True

        def __await__(self):
            async def waiter():
                if self.cancelled:
                    raise asyncio.CancelledError()
                return None  # pragma: no cover

            return waiter().__await__()

    monkeypatch.setattr("app.main.init_db", lambda: None)
    monkeypatch.setattr("app.core.database.SessionLocal", fake_session_local)
    monkeypatch.setattr(
        "app.main.settings.MONITORED_LOCATIONS",
        {"demo": {"address": "1 Demo Way", "display_name": "Demo", "icon": "📍"}},
    )

    async def no_op_geocode(db):
        return 0

    monkeypatch.setattr("app.main.GeocodingService.geocode_locations", staticmethod(no_op_geocode))
    monkeypatch.setattr("app.main.asyncio.create_task", lambda coro: FakeTask(coro))

    async def run_lifespan_once():
        async with lifespan(fastapi_app):
            return None

    asyncio.run(run_lifespan_once())
    assert fake_sessions
    assert all(session.closed for session in fake_sessions)

    monkeypatch.setattr(
        "app.main.init_db", lambda: (_ for _ in ()).throw(RuntimeError("db init failed"))
    )
    with pytest.raises(RuntimeError):
        asyncio.run(run_lifespan_once())
