"""
Comprehensive data aggregation service that unifies source-backed data.

Combines:
- Weather (NOAA, Meteostat, Open-Meteo)
- Satellite scene metadata (Sentinel-2, Landsat)
- Topographic analysis (USGS 3DEP)
- Vegetation/fuel metadata when available
- Fire data from upstream ingestion
- ML predictions from the primary v1 model stack

Provides normalized, quality-controlled data for analytics and ML models
without inventing placeholder values when a source cannot provide them.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from app.core.config import get_settings
from app.ml.anomaly_detection import AnomalyDetector
from app.ml.fire_prediction import FireSpreadPredictor
from app.ml.risk_scoring import RiskScorer
from app.services.advanced_weather import AdvancedWeatherService
from app.services.external_intelligence import ExternalIntelligenceService
from app.services.satellite_service import (
    SatelliteFireService,
    TopographicService,
    VegetationService,
)

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveFireData:
    """Complete fire dataset with optional fields when sources cannot supply them."""

    fire_id: str
    timestamp: datetime

    # Fire metrics
    fire_latitude: float
    fire_longitude: float
    acres_burned: float | None
    percent_contained: float | None
    spread_rate_mph: float | None
    direction_bearing: int | None

    # Location info
    location_id: str | None
    location_latitude: float
    location_longitude: float
    distance_miles: float

    # Weather data
    temperature_f: float | None
    humidity_pct: float | None
    wind_speed_mph: float | None
    wind_direction_deg: int | None
    pressure_mb: float | None
    fwi: float | None
    weather_source: str | None

    # Satellite data
    ndvi: float | None
    lst_kelvin: float | None
    cloud_coverage_pct: float | None
    satellite_confidence: float | None

    # Topographic data
    elevation_m: float | None
    slope_deg: float | None
    aspect_deg: int | None
    fuel_model: str | None
    fuel_load_tons_acre: float | None

    # ML predictions
    predicted_spread_rate_mph: float | None
    spread_rate_confidence: float | None
    is_anomalous: bool | None
    anomaly_score: float | None
    threat_score: float | None
    threat_level: str | None

    # Data quality
    data_completeness: float
    confidence_level: float

    # External intelligence
    incident_perimeter_distance_miles: float | None
    road_closure_count_nearby: int | None
    lightning_strikes_nearby: int | None
    elevated_lightning_risk: bool | None
    external_reliability_status: str | None
    external_intelligence: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector for downstream models with missing values as NaN."""
        return np.array(
            [
                self._numeric_or_nan(self.acres_burned),
                self._numeric_or_nan(self.percent_contained),
                self._numeric_or_nan(self.spread_rate_mph),
                self._numeric_or_nan(self.distance_miles),
                self._numeric_or_nan(self.temperature_f),
                self._numeric_or_nan(self.humidity_pct),
                self._numeric_or_nan(self.wind_speed_mph),
                self._numeric_or_nan(self.fwi),
                self._numeric_or_nan(self.elevation_m),
                self._numeric_or_nan(self.slope_deg),
                self._numeric_or_nan(self.ndvi),
                self._numeric_or_nan(self.fuel_load_tons_acre),
                self._numeric_or_nan(None if self.lst_kelvin is None else self.lst_kelvin - 273.15),
                self._numeric_or_nan(self.incident_perimeter_distance_miles),
                self._numeric_or_nan(
                    None
                    if self.road_closure_count_nearby is None
                    else float(self.road_closure_count_nearby)
                ),
                self._numeric_or_nan(
                    None
                    if self.lightning_strikes_nearby is None
                    else float(self.lightning_strikes_nearby)
                ),
                self._numeric_or_nan(
                    None
                    if self.elevated_lightning_risk is None
                    else float(self.elevated_lightning_risk)
                ),
            ],
            dtype=float,
        ).reshape(1, -1)

    @staticmethod
    def _numeric_or_nan(value: float | None) -> float:
        """Convert an optional numeric value to float or NaN."""
        return float(value) if value is not None else np.nan


class ComprehensiveDataService:
    """Master data aggregation service backed by the primary v1 model stack."""

    def __init__(self):
        """Initialize service."""
        self.weather_service = AdvancedWeatherService()
        self.satellite_service = SatelliteFireService()
        self.topo_service = TopographicService()
        self.veg_service = VegetationService()
        self.external_intelligence_service = ExternalIntelligenceService()

        self.spread_predictor = FireSpreadPredictor()
        self.anomaly_detector = AnomalyDetector()
        self.risk_scorer = RiskScorer()

        logger.info("Comprehensive data service initialized")

    @staticmethod
    def build_data_source_inventory() -> dict[str, Any]:
        """Return a code-backed inventory of implemented wildfire data sources and major gaps."""
        runtime_settings = get_settings()

        def optional_feed(
            name: str,
            *,
            configured: bool,
            surface: str,
            purpose: str,
            note: str,
        ) -> dict[str, str]:
            return {
                "name": name,
                "status": "configured" if configured else "available_not_configured",
                "surface": surface,
                "purpose": purpose,
                "note": note,
            }

        return {
            "overall_assessment": (
                "Firewatch already has solid MVP coverage across incidents, weather, smoke, "
                "satellite, "
                "and configurable ops overlays, but it is still missing several high-impact "
                "authoritative "
                "and training-truth sources needed for materially better wildfire ML."
            ),
            "implemented_now": {
                "incident_feeds": [
                    {
                        "name": "CAL FIRE incidents",
                        "status": "implemented",
                        "surface": "backend",
                        "purpose": "Authoritative California incident ingestion",
                    },
                    {
                        "name": "NASA EONET wildfire events",
                        "status": "implemented",
                        "surface": "backend",
                        "purpose": "National and global wildfire event coverage for redundancy",
                    },
                    {
                        "name": "NASA FIRMS thermal anomalies",
                        "status": "implemented",
                        "surface": "backend+edge",
                        "purpose": "Active hotspot detection and fire activity augmentation",
                    },
                ],
                "weather_and_air_quality": [
                    {
                        "name": "NOAA/NWS weather",
                        "status": "implemented",
                        "surface": "backend+edge",
                        "purpose": "Primary operational weather observations and alerts",
                    },
                    {
                        "name": "Open-Meteo weather and AQI",
                        "status": "implemented",
                        "surface": "backend+edge",
                        "purpose": "Forecast and air-quality fallback/merge source",
                    },
                    {
                        "name": "Meteostat observations",
                        "status": "implemented",
                        "surface": "backend",
                        "purpose": "Historical and blended weather context",
                    },
                    {
                        "name": "NOAA HMS smoke polygons",
                        "status": "implemented",
                        "surface": "backend",
                        "purpose": "Smoke exposure assessment from authoritative plume geometry",
                    },
                    optional_feed(
                        "EPA AirNow observations",
                        configured=bool(runtime_settings.AIRNOW_API_KEY),
                        surface="backend",
                        purpose="Higher-confidence AQI and PM2.5 situational awareness",
                        note="Merged when an AirNow API key is configured.",
                    ),
                    optional_feed(
                        "PurpleAir neighborhood sensors",
                        configured=bool(runtime_settings.PURPLEAIR_API_KEY),
                        surface="edge",
                        purpose="Hyperlocal smoke and PM2.5 context near watched locations",
                        note="Currently available through the edge intelligence overlay path.",
                    ),
                ],
                "remote_sensing_and_terrain": [
                    {
                        "name": "Sentinel-2 NDVI",
                        "status": "implemented",
                        "surface": "backend",
                        "purpose": "Vegetation vigor and burn-adjacent land cover context",
                    },
                    {
                        "name": "Landsat thermal scenes",
                        "status": "implemented",
                        "surface": "backend",
                        "purpose": "Thermal surface context and cloud-aware scene metadata",
                    },
                    {
                        "name": "USGS 3DEP elevation/topography",
                        "status": "implemented",
                        "surface": "backend",
                        "purpose": (
                            "Elevation, slope, and aspect features for risk and spread context"
                        ),
                    },
                ],
                "operational_overlays": [
                    optional_feed(
                        "Configured incident perimeter GeoJSON",
                        configured=bool(runtime_settings.FIRE_PERIMETER_GEOJSON_URL),
                        surface="backend+edge",
                        purpose="Nearest perimeter overlay with acres and containment context",
                        note=(
                            "Now consumed in backend ingestion and live threat context, but still "
                            "not a historical perimeter truth corpus for ML."
                        ),
                    ),
                    optional_feed(
                        "Configured HMS smoke GeoJSON",
                        configured=bool(runtime_settings.HMS_SMOKE_GEOJSON_URL),
                        surface="edge",
                        purpose="External intelligence smoke overlay near watched locations",
                        note="Separate from the backend HMS KML smoke assessment pipeline.",
                    ),
                    optional_feed(
                        "Configured road closures GeoJSON",
                        configured=bool(runtime_settings.ROAD_CLOSURES_GEOJSON_URL),
                        surface="backend+edge",
                        purpose="Closure-aware evacuation and route planning context",
                        note=(
                            "Now consumed in backend threat and evacuation context, but not a "
                            "historical training label source."
                        ),
                    ),
                    optional_feed(
                        "Configured utility status feed",
                        configured=bool(runtime_settings.UTILITY_STATUS_URL),
                        surface="edge",
                        purpose="PSPS and outage context for decision support",
                        note="Improves operational awareness rather than spread-physics accuracy.",
                    ),
                    optional_feed(
                        "Configured lightning nowcast feed",
                        configured=bool(runtime_settings.LIGHTNING_NOWCAST_URL),
                        surface="backend+edge",
                        purpose="Ignition-risk and rapid situational-awareness overlay",
                        note=(
                            "Now consumed in backend threat context, but is not yet tied into "
                            "model training truth."
                        ),
                    ),
                ],
            },
            "highest_impact_gaps": [
                {
                    "priority": 1,
                    "name": "Authoritative national incident metadata and perimeter history",
                    "why_it_matters": (
                        "IRWIN/InciWeb/NIFC-style incident operations data and perimeter "
                        "progression history "
                        "would improve both operational accuracy and supervised training labels."
                    ),
                },
                {
                    "priority": 2,
                    "name": "LANDFIRE fuels, canopy, and fuel loading",
                    "why_it_matters": (
                        "Real fuel models and canopy structure would add much stronger "
                        "spread-physics features "
                        "than another generic incident feed."
                    ),
                },
                {
                    "priority": 3,
                    "name": "RAWS/Mesonet observations and HRRR/RAP-class fire weather",
                    "why_it_matters": (
                        "Local station observations and higher-resolution wind/humidity forecasts "
                        "matter more "
                        "for operational fire behavior than blended public weather alone."
                    ),
                },
                {
                    "priority": 4,
                    "name": "Authoritative lightning and ignition intelligence",
                    "why_it_matters": (
                        "GOES GLM or similar lightning products help detect dry-lightning ignition "
                        "risk and "
                        "new-start conditions earlier."
                    ),
                },
                {
                    "priority": 5,
                    "name": (
                        "Historical training truth tied to incidents, perimeters, weather, and "
                        "fuels"
                    ),
                    "why_it_matters": (
                        "Real perimeter growth and containment progression labels will move "
                        "benchmark quality more "
                        "than adding more public API overlays."
                    ),
                },
            ],
            "recommended_priority_order": [
                "Authoritative incident metadata and perimeter history",
                "LANDFIRE fuels and canopy",
                "RAWS/Mesonet observations with HRRR/RAP-grade fire weather",
                "Lightning intelligence",
                "Ops layers like closures, evacuations, and utilities",
            ],
            "ml_truth_limitations": [
                "Operational overlays exist, but they do not yet form a historical training "
                "corpus.",
                "Current benchmark fixtures are useful for regression testing, not for claiming "
                "wildfire spread state of the art.",
                "The largest ML gain is likely to come from historical perimeter progression "
                "truth, not from adding another generic incident API.",
            ],
        }

    async def get_comprehensive_fire_data(
        self,
        fire_id: str,
        fire_lat: float,
        fire_lon: float,
        location_lat: float,
        location_lon: float,
        fire_data: dict[str, Any] | None = None,
        location_data: dict[str, Any] | None = None,
    ) -> ComprehensiveFireData:
        """Gather comprehensive source-backed data for a fire."""
        timestamp = datetime.now(UTC)
        fire_context = fire_data or {}
        location_context = location_data or {}

        weather_task = self._gather_weather_data(fire_lat, fire_lon)
        satellite_task = self._gather_satellite_data(fire_lat, fire_lon)
        topo_task = self._gather_topographic_data(fire_lat, fire_lon)
        veg_task = self._gather_vegetation_data(fire_lat, fire_lon)
        intelligence_task = self.external_intelligence_service.collect_for_fire_and_location(
            fire_lat,
            fire_lon,
            location_lat,
            location_lon,
        )

        (
            weather_data,
            satellite_data,
            topo_data,
            veg_data,
            external_intelligence,
        ) = await asyncio.gather(
            weather_task,
            satellite_task,
            topo_task,
            veg_task,
            intelligence_task,
            return_exceptions=True,
        )

        weather_data = weather_data if isinstance(weather_data, dict) else {}
        satellite_data = satellite_data if isinstance(satellite_data, dict) else {}
        topo_data = topo_data if isinstance(topo_data, dict) else {}
        veg_data = veg_data if isinstance(veg_data, dict) else {}
        external_intelligence = (
            external_intelligence if isinstance(external_intelligence, dict) else {}
        )
        effective_fire_context = self.external_intelligence_service.enrich_fire_context(
            fire_context,
            external_intelligence,
        )

        distance_miles = self._haversine_distance(
            fire_lat,
            fire_lon,
            location_lat,
            location_lon,
        )

        ml_data = self._get_ml_predictions(
            fire_id,
            fire_lat,
            fire_lon,
            location_lat,
            location_lon,
            distance_miles,
            effective_fire_context,
            location_context,
            external_intelligence,
        )

        completeness, confidence = self._assess_data_quality(
            weather_data,
            satellite_data,
            topo_data,
            veg_data,
        )

        incident_perimeter = (
            (external_intelligence.get("incident_perimeters") or {}).get("nearest_incident")
            if isinstance(external_intelligence, dict)
            else None
        ) or {}
        road_closures = external_intelligence.get("road_closures") or {}
        lightning_nowcast = external_intelligence.get("lightning_nowcast") or {}
        reliability = external_intelligence.get("reliability") or {}

        return ComprehensiveFireData(
            fire_id=fire_id,
            timestamp=timestamp,
            fire_latitude=fire_lat,
            fire_longitude=fire_lon,
            acres_burned=self._to_float(effective_fire_context.get("acres_burned")),
            percent_contained=self._to_float(effective_fire_context.get("percent_contained")),
            spread_rate_mph=self._to_float(effective_fire_context.get("spread_rate_mph")),
            direction_bearing=self._to_int(
                effective_fire_context.get("direction_bearing")
                or effective_fire_context.get("bearing_degrees")
            ),
            location_id=location_context.get("location_id"),
            location_latitude=location_lat,
            location_longitude=location_lon,
            distance_miles=distance_miles,
            temperature_f=self._to_float(weather_data.get("temperature_f")),
            humidity_pct=self._to_float(weather_data.get("humidity_pct")),
            wind_speed_mph=self._to_float(weather_data.get("wind_speed_mph")),
            wind_direction_deg=self._to_int(weather_data.get("wind_direction_deg")),
            pressure_mb=self._to_float(weather_data.get("pressure_mb")),
            fwi=self._to_float(weather_data.get("fwi")),
            weather_source=weather_data.get("source"),
            ndvi=self._to_float(satellite_data.get("ndvi") or veg_data.get("ndvi")),
            lst_kelvin=self._to_float(satellite_data.get("lst_kelvin")),
            cloud_coverage_pct=self._to_float(satellite_data.get("cloud_coverage_pct")),
            satellite_confidence=self._to_float(satellite_data.get("confidence")),
            elevation_m=self._to_float(topo_data.get("elevation_m")),
            slope_deg=self._to_float(topo_data.get("slope_deg")),
            aspect_deg=self._to_int(topo_data.get("aspect_deg")),
            fuel_model=veg_data.get("fuel_model"),
            fuel_load_tons_acre=self._to_float(veg_data.get("fuel_load_tons_acre")),
            predicted_spread_rate_mph=self._to_float(ml_data.get("predicted_spread_rate")),
            spread_rate_confidence=self._to_float(ml_data.get("spread_confidence")),
            is_anomalous=ml_data.get("is_anomalous"),
            anomaly_score=self._to_float(ml_data.get("anomaly_score")),
            threat_score=self._to_float(ml_data.get("threat_score")),
            threat_level=ml_data.get("threat_level"),
            data_completeness=completeness,
            confidence_level=confidence,
            incident_perimeter_distance_miles=self._to_float(
                incident_perimeter.get("distance_miles")
            ),
            road_closure_count_nearby=self._to_int(road_closures.get("closure_count_nearby")),
            lightning_strikes_nearby=self._to_int(lightning_nowcast.get("strikes_nearby_count")),
            elevated_lightning_risk=(
                bool(lightning_nowcast.get("elevated_ignition_risk")) if lightning_nowcast else None
            ),
            external_reliability_status=(
                str(reliability.get("status")) if reliability.get("status") is not None else None
            ),
            external_intelligence=external_intelligence or None,
        )

    async def _gather_weather_data(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Gather merged weather data from multiple sources."""
        try:
            weather = await self.weather_service.merge_weather_sources(latitude, longitude)
            if not weather:
                return {}

            fwi_indices = self.weather_service.calculate_fire_weather_index_v2(weather)
            return {
                "temperature_f": weather.temperature_f,
                "humidity_pct": weather.humidity_pct,
                "wind_speed_mph": weather.wind_speed_mph,
                "wind_direction_deg": weather.wind_direction_deg,
                "pressure_mb": weather.pressure_mb,
                "fwi": fwi_indices.get("fwi"),
                "source": weather.source,
            }
        except Exception as e:
            logger.warning("Weather gathering failed: %s", e)
            return {}

    async def _gather_satellite_data(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Gather source-backed satellite scene metadata."""
        try:
            sentinel_task = self.satellite_service.get_sentinel2_ndvi(latitude, longitude)
            landsat_task = self.satellite_service.get_landsat_thermal(latitude, longitude)
            firms_task = self.satellite_service.get_firms_detections(latitude, longitude)

            satellite_results: tuple[Any, ...] = await asyncio.gather(
                sentinel_task,
                landsat_task,
                firms_task,
                return_exceptions=True,
            )
            sentinel_data, landsat_data, firms_data = satellite_results

            sentinel = sentinel_data if isinstance(sentinel_data, dict) else {}
            landsat = landsat_data if isinstance(landsat_data, dict) else {}
            firms = firms_data if isinstance(firms_data, list) else []

            cloud_candidates = [
                value
                for value in (
                    self._to_float(sentinel.get("cloud_coverage")),
                    self._to_float(landsat.get("cloud_coverage")),
                )
                if value is not None
            ]

            result: dict[str, Any] = {}
            if sentinel.get("ndvi_mean") is not None:
                result["ndvi"] = sentinel.get("ndvi_mean")
            if landsat.get("land_surface_temp_k") is not None:
                result["lst_kelvin"] = landsat.get("land_surface_temp_k")

            if cloud_candidates:
                result["cloud_coverage_pct"] = min(cloud_candidates)
                result["confidence"] = max(0.0, 1.0 - (result["cloud_coverage_pct"] / 100.0))

            if firms:
                result["fire_detection_count"] = len(firms)
                radiative_power = [
                    value
                    for detection in firms
                    if (value := self._to_float(getattr(detection, "radiative_power", None)))
                    is not None
                ]
                if radiative_power:
                    result["mean_radiative_power_mw"] = float(np.mean(radiative_power))

                max_detection_confidence = (
                    max(float(getattr(detection, "confidence", 0.0)) for detection in firms) / 100.0
                )
                result["confidence"] = max(
                    self._to_float(result.get("confidence")) or 0.0,
                    max_detection_confidence,
                )

            sources = [
                source for source in (sentinel.get("source"), landsat.get("source")) if source
            ]
            sensors = {str(getattr(detection, "sensor", None) or "") for detection in firms}
            sources.extend(f"FIRMS:{sensor}" for sensor in sorted(sensors - {""}))
            if sources:
                result["source"] = "+".join(sources)

            scene_ids = [
                scene_id
                for scene_id in (sentinel.get("scene_id"), landsat.get("scene_id"))
                if scene_id
            ]
            if scene_ids:
                result["scene_ids"] = scene_ids

            return result
        except Exception as e:
            logger.warning("Satellite gathering failed: %s", e)
            return {}

    async def _gather_topographic_data(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Gather topographic data without fabricating unavailable fields."""
        try:
            elev_data = await self.topo_service.get_elevation_and_slope(latitude, longitude)
            if not elev_data:
                return {}

            return {
                "elevation_m": elev_data.get("elevation_m"),
                "slope_deg": elev_data.get("slope_deg"),
                "aspect_deg": elev_data.get("aspect_deg"),
                "source": elev_data.get("source"),
            }
        except Exception as e:
            logger.warning("Topographic gathering failed: %s", e)
            return {}

    async def _gather_vegetation_data(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Gather vegetation and fuel metadata when an upstream source can supply it."""
        try:
            ndvi_task = self.veg_service.get_ndvi(latitude, longitude)
            moisture_task = self.veg_service.get_moisture_stress(latitude, longitude)
            fuel_task = self.topo_service.get_fuel_type(latitude, longitude)

            vegetation_results: tuple[Any, ...] = await asyncio.gather(
                ndvi_task,
                moisture_task,
                fuel_task,
                return_exceptions=True,
            )

            result: dict[str, Any] = {}
            for dataset in vegetation_results:
                if isinstance(dataset, dict):
                    for key, value in dataset.items():
                        if value is not None:
                            result[key] = value

            return result
        except Exception as e:
            logger.warning("Vegetation gathering failed: %s", e)
            return {}

    def _get_ml_predictions(
        self,
        fire_id: str,
        fire_lat: float,
        fire_lon: float,
        location_lat: float,
        location_lon: float,
        distance_miles: float,
        fire_data: dict[str, Any],
        location_data: dict[str, Any],
        external_intelligence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get ML predictions from the primary v1 model stack when supported by inputs."""
        try:
            history = self._coerce_distance_history(
                fire_data.get("distance_history") or fire_data.get("history") or []
            )
            intelligence_adjustments = (
                self.external_intelligence_service.summarize_threat_modifiers(
                    external_intelligence,
                )
            )

            fire = self._build_fire_proxy(fire_id, fire_lat, fire_lon, fire_data)
            location = self._build_location_proxy(location_lat, location_lon, location_data)

            predicted_spread_rate: float | None = None
            spread_confidence: float | None = None
            if len(history) >= 2:
                predicted_spread_rate = self.spread_predictor.predict_spread_rate(
                    cast(Any, fire),
                    cast(Any, history),
                )
                spread_confidence = min(0.95, 0.5 + 0.05 * len(history))
            elif fire_data.get("spread_rate_mph") is not None:
                predicted_spread_rate = self._to_float(fire_data.get("spread_rate_mph"))
                spread_confidence = 0.60

            is_anomalous: bool | None = None
            anomaly_score: float | None = None
            if len(history) >= 2:
                anomaly_result = self.anomaly_detector.detect_anomaly(
                    cast(Any, fire),
                    cast(Any, history),
                )
                is_anomalous = anomaly_result.get("is_anomaly")
                anomaly_score = self._to_float(anomaly_result.get("anomaly_score"))

            threat_score: float | None = None
            threat_level: str | None = None
            if (
                fire_data.get("acres_burned") is not None
                or fire_data.get("percent_contained") is not None
            ):
                threat_score = self.risk_scorer.calculate_combined_threat_score(
                    cast(Any, location),
                    cast(Any, fire),
                    distance_miles,
                    predicted_spread_rate,
                )
                threat_score = min(
                    100.0,
                    float(threat_score) + float(intelligence_adjustments.get("score_boost") or 0.0),
                )
                threat_level = self.risk_scorer.get_threat_description(threat_score)

            return {
                "predicted_spread_rate": predicted_spread_rate,
                "spread_confidence": spread_confidence,
                "is_anomalous": is_anomalous,
                "anomaly_score": anomaly_score,
                "threat_score": threat_score,
                "threat_level": threat_level,
                "external_risk_modifiers": intelligence_adjustments.get("modifiers") or [],
                "external_risk_score_boost": intelligence_adjustments.get("score_boost") or 0.0,
            }
        except Exception as e:
            logger.warning("ML predictions failed: %s", e)
            return {
                "predicted_spread_rate": None,
                "spread_confidence": None,
                "is_anomalous": None,
                "anomaly_score": None,
                "threat_score": None,
                "threat_level": None,
                "external_risk_modifiers": [],
                "external_risk_score_boost": 0.0,
            }

    @staticmethod
    def _build_fire_proxy(
        fire_id: str,
        fire_lat: float,
        fire_lon: float,
        fire_data: dict[str, Any],
    ) -> SimpleNamespace:
        """Build a lightweight fire object for the v1 ML stack."""
        return SimpleNamespace(
            fire_id=fire_id,
            name=fire_data.get("name", fire_id),
            latitude=fire_lat,
            longitude=fire_lon,
            acres_burned=float(fire_data.get("acres_burned") or 0.0),
            percent_contained=fire_data.get("percent_contained"),
            discovered_at=ComprehensiveDataService._parse_datetime(fire_data.get("discovered_at")),
            last_updated=ComprehensiveDataService._parse_datetime(fire_data.get("last_updated"))
            or datetime.now(UTC),
            bearing_degrees=fire_data.get("bearing_degrees") or fire_data.get("direction_bearing"),
            growth_acres_24h=fire_data.get("growth_acres_24h"),
            extra_metadata={
                **(fire_data.get("extra_metadata") or {}),
                **(
                    {"growth_acres_24h": fire_data.get("growth_acres_24h")}
                    if fire_data.get("growth_acres_24h") is not None
                    else {}
                ),
            },
        )

    @staticmethod
    def _build_location_proxy(
        location_lat: float,
        location_lon: float,
        location_data: dict[str, Any],
    ) -> SimpleNamespace:
        """Build a lightweight location object for the v1 ML stack."""
        return SimpleNamespace(
            location_id=location_data.get("location_id"),
            latitude=location_lat,
            longitude=location_lon,
        )

    @classmethod
    def _coerce_distance_history(cls, raw_history: Any) -> list[SimpleNamespace]:
        """Normalize incoming history into lightweight objects with expected fields."""
        if not isinstance(raw_history, list):
            return []

        history: list[SimpleNamespace] = []
        for item in raw_history:
            if isinstance(item, dict):
                timestamp = cls._parse_datetime(item.get("timestamp") or item.get("recorded_at"))
                distance = cls._to_float(item.get("distance_miles"))
                bearing = cls._to_float(item.get("bearing_degrees") or item.get("bearing"))
            else:
                timestamp = cls._parse_datetime(
                    getattr(item, "timestamp", getattr(item, "recorded_at", None))
                )
                distance = cls._to_float(getattr(item, "distance_miles", None))
                bearing = cls._to_float(getattr(item, "bearing_degrees", None))

            if timestamp is None or distance is None:
                continue

            history.append(
                SimpleNamespace(
                    timestamp=timestamp,
                    distance_miles=distance,
                    bearing_degrees=bearing or 0.0,
                )
            )

        return history

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        """Parse an incoming datetime-like value to a datetime instance."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Convert value to float when possible."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: Any) -> int | None:
        """Convert value to int when possible."""
        if value is None:
            return None
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in miles."""
        R = 3959  # Earth radius in miles

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        return float(R * c)

    @staticmethod
    def _assess_data_quality(
        weather_data: dict[str, Any],
        satellite_data: dict[str, Any],
        topo_data: dict[str, Any],
        veg_data: dict[str, Any],
    ) -> tuple[float, float]:
        """Assess data quality and completeness."""
        completeness_scores = []
        confidence_scores = []

        weather_keys = ["temperature_f", "humidity_pct", "wind_speed_mph", "pressure_mb", "fwi"]
        weather_complete = sum(
            1 for key in weather_keys if weather_data.get(key) is not None
        ) / len(weather_keys)
        completeness_scores.append(weather_complete)
        weather_source = str(weather_data.get("source") or "")
        if "NOAA" in weather_source:
            confidence_scores.append(0.85)
        elif weather_source:
            confidence_scores.append(0.70)
        else:
            confidence_scores.append(0.0)

        sat_keys = ["ndvi", "lst_kelvin", "cloud_coverage_pct"]
        sat_complete = sum(1 for key in sat_keys if satellite_data.get(key) is not None) / len(
            sat_keys
        )
        completeness_scores.append(sat_complete)
        sat_confidence = ComprehensiveDataService._to_float(satellite_data.get("confidence"))
        if sat_confidence is not None:
            confidence_scores.append(sat_confidence)
        elif satellite_data.get("cloud_coverage_pct") is not None:
            confidence_scores.append(
                min(1.0, 1.0 - (float(satellite_data["cloud_coverage_pct"]) / 100.0))
            )
        else:
            confidence_scores.append(0.0)

        topo_keys = ["elevation_m", "slope_deg", "aspect_deg"]
        topo_complete = sum(1 for key in topo_keys if topo_data.get(key) is not None) / len(
            topo_keys
        )
        completeness_scores.append(topo_complete)
        confidence_scores.append(0.9 if topo_data.get("elevation_m") is not None else 0.0)

        veg_keys = ["fuel_model", "fuel_load_tons_acre", "moisture_content_pct"]
        veg_complete = sum(1 for key in veg_keys if veg_data.get(key) is not None) / len(veg_keys)
        completeness_scores.append(veg_complete)
        confidence_scores.append(0.75 if veg_complete > 0 else 0.0)

        completeness = float(np.mean(completeness_scores) * 100)
        confidence = float(np.mean(confidence_scores) * 100)

        return completeness, confidence
