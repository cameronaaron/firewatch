import asyncio
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
from shapely.geometry import Point, Polygon

from app.models.database import Fire, Location
from app.services.advanced_weather import AdvancedWeatherService, WeatherData
from app.services.air_quality import AirQualityService
from app.services.fire_data import FireDataService
from app.services.ml_analysis import MLFireAnalysisService
from app.services.ml_analysis import settings as ml_settings
from app.services.satellite_service import FireDetection
from app.services.smoke_data import SmokeDataService, SmokePlume


class DummyAsyncResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_advanced_weather_source_fetch_paths(monkeypatch):
    service = AdvancedWeatherService()

    class OkClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, **kwargs):
            if "api.weather.gov/points" in url:
                return DummyAsyncResponse(
                    200,
                    {"properties": {"forecast": "https://forecast.example/noaa"}},
                )
            if "forecast.example/noaa" in url:
                return DummyAsyncResponse(
                    200,
                    {
                        "properties": {
                            "periods": [
                                {
                                    "temperature": 87,
                                    "windSpeed": "14 mph",
                                    "windDirection": "NW",
                                    "shortForecast": "Dry",
                                    "detailedForecast": "Clear skies",
                                    "hazards": [],
                                }
                            ]
                        }
                    },
                )
            if "open-meteo.com" in url and kwargs.get("params", {}).get("current"):
                return DummyAsyncResponse(
                    200,
                    {
                        "current": {
                            "temperature_2m": 82.0,
                            "relative_humidity_2m": 25.0,
                            "wind_speed_10m": 12.0,
                            "wind_direction_10m": 180,
                            "precipitation": 0.0,
                            "surface_pressure": 1008.0,
                            "dew_point_2m": 45.0,
                            "visibility": 8046.72,
                            "cloud_cover": 10,
                            "time": "2026-05-21T10:00",
                        }
                    },
                )
            if "meteostat.net" in url:
                return DummyAsyncResponse(
                    200,
                    {
                        "data": {
                            "temp": 25.0,
                            "rhum": 35,
                            "wspd": 10,
                            "wdir": 200,
                            "prcp": 2.54,
                            "pres": 1012,
                            "dwpt": 10,
                            "vvis": 1609.34,
                        }
                    },
                )
            if "open-meteo.com" in url and kwargs.get("params", {}).get("daily"):
                return DummyAsyncResponse(
                    200,
                    {
                        "daily": {
                            "time": ["2026-05-21", "2026-05-22"],
                            "temperature_2m_max": [90, 91],
                            "temperature_2m_min": [60, 61],
                            "precipitation_sum": [0.0, 0.1],
                            "wind_speed_10m_max": [12, 14],
                            "relative_humidity_2m_mean": [20, 22],
                        }
                    },
                )
            return DummyAsyncResponse(500, {})

    monkeypatch.setattr("app.services.advanced_weather.httpx.AsyncClient", lambda: OkClient())

    noaa = asyncio.run(service.get_noaa_weather(34.0, -118.0))
    openmeteo = asyncio.run(service.get_openmeteo_weather(34.0, -118.0))
    meteostat = asyncio.run(service.get_meteostat_weather(34.0, -118.0))
    forecast = asyncio.run(service.get_forecast(34.0, -118.0, days=2))

    assert noaa["wind_direction_deg"] == 315
    assert openmeteo["visibility_miles"] == pytest.approx(5.0, abs=1e-3)
    assert meteostat["temperature_f"] == pytest.approx(77.0)
    assert len(forecast["forecast"]) == 2

    class ErrorClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            raise RuntimeError("network")

    monkeypatch.setattr("app.services.advanced_weather.httpx.AsyncClient", lambda: ErrorClient())
    assert asyncio.run(service.get_noaa_weather(34.0, -118.0)) is None
    assert asyncio.run(service.get_openmeteo_weather(34.0, -118.0)) is None
    assert asyncio.run(service.get_meteostat_weather(34.0, -118.0)) is None
    assert asyncio.run(service.get_forecast(34.0, -118.0, days=2)) is None


def test_advanced_weather_indices_and_helpers_cover_edges(monkeypatch):
    service = AdvancedWeatherService()

    assert (
        service.calculate_fire_weather_index_v2(
            WeatherData(
                timestamp=datetime.now(UTC),
                temperature_f=None,
                humidity_pct=40,
                wind_speed_mph=8,
                wind_direction_deg=90,
                precipitation_in=0.0,
                pressure_mb=None,
                dewpoint_f=None,
                visibility_miles=None,
                cloud_coverage_pct=None,
            )
        )["danger_level"]
        is None
    )

    full = service.calculate_fire_weather_index_v2(
        WeatherData(
            timestamp=datetime.now(UTC),
            temperature_f=102,
            humidity_pct=12,
            wind_speed_mph=24,
            wind_direction_deg=180,
            precipitation_in=0.0,
            pressure_mb=1000,
            dewpoint_f=45,
            visibility_miles=10,
            cloud_coverage_pct=5,
        )
    )
    assert full["danger_level"] in {"HIGH", "VERY_HIGH", "EXTREME"}

    assert (
        AdvancedWeatherService._blend_numeric(
            [{"source": "A", "temperature_f": 10}], "temperature_f", {"A": 0.0}
        )
        is None
    )
    assert (
        AdvancedWeatherService._preferred_direction(
            [
                {"source": "x", "wind_direction": "SSW"},
                {"source": "y", "wind_direction_deg": 90},
            ]
        )
        == 202
    )
    assert AdvancedWeatherService._optional_int(None) is None
    assert AdvancedWeatherService._optional_int(7.6) == 8

    async def no_sources(latitude, longitude):
        return None

    monkeypatch.setattr(service, "merge_weather_sources", no_sources)
    assert asyncio.run(service.get_current_weather(34.0, -118.0)) is None
    assert asyncio.run(service.get_fire_weather_indices(34.0, -118.0)) is None


def test_smoke_data_loading_parsing_and_assessment(monkeypatch):
    service = SmokeDataService()
    SmokeDataService._document_cache.clear()

    class DummySyncResponse:
        def __init__(self, status_code=200, text="", raise_error=None):
            self.status_code = status_code
            self.text = text
            self._raise_error = raise_error

        def raise_for_status(self):
            if self._raise_error:
                raise self._raise_error

    class Client404:
        def __init__(self, timeout=None):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            return DummySyncResponse(status_code=404)

    monkeypatch.setattr("app.services.smoke_data.httpx.Client", Client404)
    assert service._load_plumes_for_date(date.today()) == []

    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(500, request=request)

    class ClientError(Client404):
        def get(self, url):
            return DummySyncResponse(
                status_code=500,
                raise_error=httpx.HTTPStatusError("bad", request=request, response=response),
            )

    monkeypatch.setattr("app.services.smoke_data.httpx.Client", ClientError)
    assert service._load_plumes_for_date(date.today() - timedelta(days=1)) == []

    kml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <kml xmlns=\"http://www.opengis.net/kml/2.2\"><Document><Placemark>
      <styleUrl>#Smoke_Light</styleUrl>
      <description>Start Time: 2026140 1200UTC End Time: 2026140 1800UTC Density: Light Satellite: NOAA20</description>
      <Polygon><outerBoundaryIs><LinearRing><coordinates>
        -118.20,34.00,0 -118.00,34.00,0 -118.00,34.20,0 -118.20,34.20,0 -118.20,34.00,0
      </coordinates></LinearRing></outerBoundaryIs></Polygon>
    </Placemark></Document></kml>
    """

    calls = {"count": 0}

    class ClientOk(Client404):
        def get(self, url):
            calls["count"] += 1
            return DummySyncResponse(status_code=200, text=kml)

    monkeypatch.setattr("app.services.smoke_data.httpx.Client", ClientOk)
    target_date = date.today() - timedelta(days=2)
    parsed_once = service._load_plumes_for_date(target_date)
    parsed_twice = service._load_plumes_for_date(target_date)
    assert len(parsed_once) == 1
    assert len(parsed_twice) == 1
    assert calls["count"] == 1

    density, start_time, end_time, satellite = service._parse_description("n/a", "#Smoke_Heavy")
    assert density == "HEAVY"
    assert start_time is None and end_time is None and satellite is None

    assert service._polygon_from_coordinates("bad,bad") is None
    polygon = service._polygon_from_coordinates(
        "-118.1,34.0,0 -118.0,34.0,0 -118.0,34.1,0 -118.1,34.0,0"
    )
    assert polygon is not None

    inside_distance = service._distance_to_geometry_miles(
        Point(-118.1, 34.1),
        Polygon([(-118.2, 34.0), (-118.0, 34.0), (-118.0, 34.2), (-118.2, 34.2)]),
    )
    assert inside_distance == 0.0
    assert service._haversine_distance(34.0, -118.0, 34.0, -118.0) == 0.0

    now = datetime.now(UTC)

    def fake_load_plumes_for_date(_target):
        return [
            SmokePlume(
                density="HEAVY",
                geometry=Polygon([(-118.2, 34.0), (-118.0, 34.0), (-118.0, 34.2), (-118.2, 34.2)]),
                start_time=now - timedelta(hours=3),
                end_time=now - timedelta(hours=1),
                satellite="SNPP",
            ),
            SmokePlume(
                density="LIGHT",
                geometry=Polygon([(-119.0, 35.0), (-118.8, 35.0), (-118.8, 35.2), (-119.0, 35.2)]),
                start_time=now - timedelta(hours=2),
                end_time=now - timedelta(hours=1),
                satellite="NOAA20",
            ),
        ]

    monkeypatch.setattr(service, "_load_plumes_for_date", fake_load_plumes_for_date)
    assessment = service.get_smoke_assessment(34.1, -118.1, lookback_days=1, max_distance_miles=50)
    assert assessment["smoke_present"] is True
    assert assessment["smoke_scope"] == "overhead"
    assert assessment["smoke_density"] == "Heavy"


def test_air_quality_branches_and_utilities(monkeypatch):
    service = AirQualityService()

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class DummyClient:
        def __init__(self, responses):
            self.responses = responses

        def get(self, url, params=None):
            if "air-quality-api" in url:
                return DummyResponse(self.responses["openmeteo"])
            return DummyResponse(self.responses["airnow"])

    openmeteo_none = service._fetch_openmeteo(
        DummyClient({"openmeteo": {"current": {}}, "airnow": []}),
        34.0,
        -118.0,
    )
    assert openmeteo_none is None

    service.airnow_api_key = "test-key"
    airnow = service._fetch_airnow(
        DummyClient(
            {
                "openmeteo": {},
                "airnow": [
                    "bad-row",
                    {"AQI": 88, "ParameterName": "O3"},
                    {
                        "AQI": 105,
                        "ParameterName": "PM2.5",
                        "Concentration": 37.1,
                        "DateObserved": "2026-05-21",
                        "HourObserved": 15,
                    },
                ],
            }
        ),
        34.0,
        -118.0,
    )
    assert airnow["air_quality_index"] == 105
    assert airnow["pm25_ug_m3"] == 37.1

    assert service._weighted_average([(None, 1.0), ("bad", 1.0), (10, 0)]) is None
    assert service._aqi_category(None) is None
    assert service._aqi_category(350) == "Hazardous"
    assert service._parse_observed_at("", 0) is None
    assert service._parse_observed_at("bad", 0) is None
    assert service._parse_airnow_observed_at({"DateObserved": "bad", "HourObserved": "x"}) is None

    old_time = datetime.now(UTC) - timedelta(hours=30)
    assert (
        service._source_weight({"source_confidence": 0.8, "air_quality_observed_at": old_time})
        < 0.8
    )
    assert (
        service._source_weight({"source_confidence": 0.8, "air_quality_observed_at": "bad"}) < 0.8
    )

    class ErrorClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, *args, **kwargs):
            raise httpx.HTTPError("boom")

    monkeypatch.setattr("app.services.air_quality.httpx.Client", ErrorClient)
    assert service.get_air_quality_assessment(34.0, -118.0) is None


def test_fire_data_source_helpers_and_merge_paths():
    assert FireDataService._latest_point_geometry("bad") is None
    assert FireDataService._latest_point_geometry([]) is None
    latest = FireDataService._latest_point_geometry(
        [
            {"type": "Point", "date": "2026-01-01", "coordinates": [0, 0]},
            {"type": "Point", "date": "2026-02-01", "coordinates": [1, 1]},
            {"type": "Polygon", "date": "2026-03-01", "coordinates": []},
        ]
    )
    assert latest["coordinates"] == [1, 1]

    assert FireDataService._source_family("CAL_FIRE_FEED") == "CAL_FIRE"
    assert FireDataService._source_family("FIRMS:VIIRS") == "FIRMS"
    assert FireDataService._source_family("EONET") == "EONET"
    assert (
        FireDataService._record_merge_key({"Name": "X", "Latitude": "bad", "Longitude": "bad"})
        == "x:0.0:0.0"
    )

    merged = {
        "a": {"Name": "Known", "Latitude": 34.0, "Longitude": -118.0, "Source": "CAL_FIRE"},
        "b": {"Name": "Hotspot", "Latitude": 34.01, "Longitude": -118.01, "Source": "FIRMS:SNPP"},
    }
    key = FireDataService._find_spatial_merge_key(
        {"Name": "Incoming", "Latitude": 34.0, "Longitude": -118.01, "Source": "FIRMS:NOAA21"},
        merged,
    )
    assert key == "a"

    combined = FireDataService._combine_source_records(
        {
            "Name": "Primary",
            "UniqueId": "p",
            "Latitude": 34.0,
            "Longitude": -118.0,
            "AcresBurned": 0.0,
            "PercentContained": 0.0,
            "Source": "EONET",
        },
        {
            "Name": "Secondary",
            "UniqueId": "s",
            "Latitude": 34.1,
            "Longitude": -118.1,
            "AcresBurned": 123.0,
            "PercentContained": 45.0,
            "Source": "CAL_FIRE",
            "ObservedAt": "2026-05-21T10:00:00Z",
            "Confidence": 90,
        },
    )
    assert combined["AcresBurned"] == 123.0
    assert "CAL_FIRE" in combined["Source"]
    assert "EONET" in combined["Source"]

    normalized = FireDataService._normalize_firms_detection(
        FireDetection(
            latitude=34.1,
            longitude=-118.2,
            confidence=85,
            sensor="viirs",
            timestamp=datetime(2026, 5, 21, 10, 0, tzinfo=UTC),
            brightness_temp=320.0,
            radiative_power=12.4,
        )
    )
    assert normalized["Source"].startswith("FIRMS")
    assert normalized["Name"].endswith("satellite hotspot")


def test_fire_data_async_source_validation_errors():
    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class DummyClient:
        def __init__(self, payload):
            self.payload = payload

        async def get(self, *args, **kwargs):
            return DummyResponse(self.payload)

    with pytest.raises(ValueError):
        asyncio.run(FireDataService._fetch_calfire_fires(DummyClient({"not": "a-list"})))

    with pytest.raises(ValueError):
        asyncio.run(FireDataService._fetch_eonet_fires(DummyClient([])))

    with pytest.raises(ValueError):
        asyncio.run(FireDataService._fetch_eonet_fires(DummyClient({"events": "bad"})))


def test_ml_analysis_bootstrap_override_and_model_selection(monkeypatch, tmp_path):
    service = MLFireAnalysisService()
    service._promoted_model_path = str(tmp_path / "promoted.pkl")

    unchanged = service._apply_bootstrap_promotion_override({"promote": True, "checks": []})
    assert unchanged["promote"] is True

    monkeypatch.setattr(
        ml_settings, "SPREAD_MODEL_BOOTSTRAP_PROMOTION_ENABLED", False, raising=False
    )
    disabled = service._apply_bootstrap_promotion_override({"promote": False, "checks": []})
    assert disabled["promote"] is False

    monkeypatch.setattr(
        ml_settings, "SPREAD_MODEL_BOOTSTRAP_PROMOTION_ENABLED", True, raising=False
    )
    (tmp_path / "promoted.pkl").write_text("artifact", encoding="utf-8")
    existing = service._apply_bootstrap_promotion_override({"promote": False, "checks": []})
    assert existing["bootstrap_override_applied"] is False

    service._promoted_model_path = str(tmp_path / "missing.pkl")
    failed = service._apply_bootstrap_promotion_override(
        {
            "promote": False,
            "checks": [
                {
                    "name": "dataset_informative",
                    "passed": False,
                    "failure_reason": "not informative",
                },
            ],
        }
    )
    assert failed["bootstrap_override_applied"] is False

    passed = service._apply_bootstrap_promotion_override(
        {
            "promote": False,
            "checks": [
                {"name": "dataset_informative", "passed": True},
                {"name": "non_tie_uplift", "passed": True},
                {"name": "minimum_group_count", "passed": True},
                {"name": "maximum_mae_std_ratio", "passed": True},
                {"name": "minimum_stability_score", "passed": True},
                {"name": "production_ready_scorecard", "passed": True},
            ],
        }
    )
    assert passed["promote"] is True
    assert passed["bootstrap_override_applied"] is True

    monkeypatch.setattr(ml_settings, "SPREAD_MODEL_RUNTIME_ENSEMBLE_ENABLED", False, raising=False)
    assert MLFireAnalysisService._select_serving_model({"best_model": "random_forest"}) == (
        "random_forest",
        "newly_promoted",
    )

    monkeypatch.setattr(ml_settings, "SPREAD_MODEL_RUNTIME_ENSEMBLE_ENABLED", True, raising=False)
    monkeypatch.setattr(
        ml_settings, "SPREAD_MODEL_RUNTIME_ENSEMBLE_MAX_MAE_DELTA_PCT", 5.0, raising=False
    )
    winner, source = MLFireAnalysisService._select_serving_model(
        {
            "best_model": "xgboost",
            "results": [
                {"name": "xgboost", "mae": 1.00},
                {"name": "voting_ensemble", "mae": 1.03},
            ],
        }
    )
    assert winner == "voting_ensemble"
    assert source == "newly_promoted_runtime_ensemble"

    loc = Location(
        location_id="loc",
        address="x",
        display_name="x",
        icon="x",
        latitude=None,
        longitude=None,
        is_active=True,
    )
    fire = Fire(
        fire_id="f", name="f", latitude=34.0, longitude=-118.0, acres_burned=1, percent_contained=1
    )
    assert MLFireAnalysisService._calculate_distance_miles(fire, loc) == 0.0
