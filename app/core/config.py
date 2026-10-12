"""Settings read from the environment and .env; unknown keys are rejected."""

import logging
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "Firewatch"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False

    # Database
    DATABASE_URL: str = "sqlite:///./firewatch.db"
    DATABASE_ECHO: bool = False  # Set to True to see SQL queries in logs

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["*"]  # In production, specify exact origins
    ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1", "testserver"]

    # Fire Data API
    CALFIRE_API_URL: str = "https://incidents.fire.ca.gov/umbraco/api/IncidentApi/List"
    FIRE_DATA_FETCH_INTERVAL: int = 600  # 10 minutes in seconds
    FIRE_DATA_TIMEOUT: int = 10
    MAX_FIRE_DISTANCE_MILES: float = 300.0
    # A fire unreported by successful fetches for this long is treated as out.
    FIRE_STALE_HOURS: float = 6.0

    # Satellite and remote sensing
    FIRMS_MAP_KEY: str | None = None
    FIRMS_API_TIMEOUT: int = 20
    FIRMS_SOURCE_IDS: list[str] = [
        "VIIRS_NOAA21_NRT",
        "VIIRS_NOAA20_NRT",
        "VIIRS_SNPP_NRT",
        "MODIS_NRT",
    ]
    SATELLITE_SAMPLE_RADIUS_PIXELS: int = 16

    # Smoke and air quality situational awareness
    HMS_SMOKE_KML_BASE_URL: str = (
        "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/KML"
    )
    HMS_SMOKE_LOOKBACK_DAYS: int = 3
    HMS_SMOKE_TIMEOUT: int = 20
    HMS_SMOKE_MAX_DISTANCE_MILES: float = 25.0
    AIR_QUALITY_BASE_URL: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    AIR_QUALITY_TIMEOUT: int = 10
    AIRNOW_API_KEY: str | None = None
    AIRNOW_BASE_URL: str = "https://www.airnowapi.org/aq/observation/latLong/current/"
    AIRNOW_RADIUS_MILES: int = 25

    # Optional external intelligence overlays shared with edge intelligence and inventory reporting
    FIRE_PERIMETER_GEOJSON_URL: str | None = None
    HMS_SMOKE_GEOJSON_URL: str | None = None
    PURPLEAIR_API_KEY: str | None = None
    ROAD_CLOSURES_GEOJSON_URL: str | None = None
    UTILITY_STATUS_URL: str | None = None
    LIGHTNING_NOWCAST_URL: str | None = None

    # Benchmark-driven spread model promotion
    HISTORICAL_BENCHMARK_MIN_SAMPLES: int = 6
    HISTORICAL_BENCHMARK_TIME_SPLIT_HOLDOUT_FRACTION: float = 0.25
    HISTORICAL_BENCHMARK_RUN_TUNING: bool = True
    SPREAD_MODEL_PROMOTION_MIN_IMPROVEMENT_PCT: float = 1.0
    SPREAD_MODEL_PROMOTION_REQUIRE_INFORMATIVE: bool = True
    SPREAD_MODEL_PROMOTION_REQUIRE_NON_TIE_UPLIFT: bool = True
    SPREAD_MODEL_PROMOTION_REQUIRE_TIME_SPLIT: bool = True
    SPREAD_MODEL_PROMOTION_REQUIRE_PRODUCTION_READY: bool = True
    SPREAD_MODEL_PROMOTION_MIN_TIME_SPLIT_IMPROVEMENT_PCT: float = 0.0
    SPREAD_MODEL_PROMOTION_MAX_TIME_SPLIT_DEGRADATION_PCT: float = 35.0
    SPREAD_MODEL_PROMOTION_MIN_GROUPS: int = 4
    SPREAD_MODEL_PROMOTION_MIN_R2: float = 0.0
    SPREAD_MODEL_PROMOTION_MAX_MAE_STD_RATIO: float = 0.65
    SPREAD_MODEL_PROMOTION_MIN_STABILITY_SCORE: float = 0.6
    SPREAD_MODEL_PRODUCTION_READY_MIN_CV_IMPROVEMENT_PCT: float = 2.0
    SPREAD_MODEL_PRODUCTION_READY_MIN_TIME_SPLIT_IMPROVEMENT_PCT: float = 1.0
    SPREAD_MODEL_PRODUCTION_READY_MIN_TIME_SPLIT_TEST_R2: float = 0.05
    SPREAD_MODEL_PRODUCTION_READY_MAX_GENERALIZATION_GAP_PCT: float = 150.0
    SPREAD_MODEL_STATE_OF_THE_ART_MIN_CV_IMPROVEMENT_PCT: float = 5.0
    SPREAD_MODEL_STATE_OF_THE_ART_MIN_CV_R2: float = 0.2
    SPREAD_MODEL_STATE_OF_THE_ART_MIN_TIME_SPLIT_IMPROVEMENT_PCT: float = 8.0
    SPREAD_MODEL_STATE_OF_THE_ART_MIN_TIME_SPLIT_TEST_R2: float = 0.35
    SPREAD_MODEL_STATE_OF_THE_ART_MAX_GENERALIZATION_GAP_PCT: float = 80.0
    SPREAD_MODEL_PROMOTED_PATH: str = "./artifacts/promoted_spread_model.joblib"
    SPREAD_MODEL_BOOTSTRAP_PROMOTION_ENABLED: bool = False
    SPREAD_MODEL_RUNTIME_ENSEMBLE_ENABLED: bool = False
    SPREAD_MODEL_RUNTIME_ENSEMBLE_MAX_MAE_DELTA_PCT: float = 3.0

    # Alert Configuration
    EVACUATION_DISTANCE_MONITOR: float = 5.0
    EVACUATION_DISTANCE_PREPARE: float = 3.0
    EVACUATION_DISTANCE_WARNING: float = 2.0
    EVACUATION_DISTANCE_EVACUATE: float = 1.0

    # Optional boot-time seed data. Leave empty in production so user watches stay client-managed.
    MONITORED_LOCATIONS: dict[str, dict[str, str]] = {}

    # Geocoding
    GEOCODER_USER_AGENT: str = "firewatch-app"
    GEOCODER_TIMEOUT: int = 10
    CENSUS_GEOCODER_URL: str = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    CENSUS_GEOCODER_BENCHMARK: str = "Public_AR_Current"

    # Caching
    CACHE_TTL_SECONDS: int = 300  # 5 minutes
    HISTORICAL_DATA_RETENTION_HOURS: int = 24

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


@lru_cache
def get_settings() -> Settings:
    """Get settings instance (cached)."""
    return Settings()


if __name__ == "__main__":  # pragma: no cover
    settings = get_settings()
    print(f"App: {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Database: {settings.DATABASE_URL}")
