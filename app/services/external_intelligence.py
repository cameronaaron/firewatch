"""Backend external intelligence feed fetching shared across ingestion and ML."""

import asyncio
import logging
import math
import re
from typing import Any

import httpx

from app.core.config import get_settings
from app.models.database import Location

logger = logging.getLogger(__name__)
settings = get_settings()


class ExternalIntelligenceService:
    """Fetch configured perimeter, road-closure, and lightning intelligence feeds."""

    async def fetch_incident_perimeter_features(self) -> list[dict[str, Any]]:
        """Fetch configured incident perimeter GeoJSON features."""
        if not settings.FIRE_PERIMETER_GEOJSON_URL:
            return []
        return await self._fetch_geojson_features(settings.FIRE_PERIMETER_GEOJSON_URL)

    async def build_perimeter_fire_records(
        self,
        locations: list[Location],
    ) -> list[dict[str, Any]]:
        """Convert configured perimeter features into normalized fire records near watches."""
        features = await self.fetch_incident_perimeter_features()
        candidate_locations = [
            location
            for location in locations
            if location.latitude is not None and location.longitude is not None
        ]
        if not features or not candidate_locations:
            return []

        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for feature in features:
            geometry = feature.get("geometry")
            reference_point = self._feature_reference_point(feature)
            if reference_point is None:
                continue

            latitude, longitude = reference_point
            minimum_distance = min(
                self.haversine_distance(
                    latitude,
                    longitude,
                    float(location.latitude),
                    float(location.longitude),
                )
                for location in candidate_locations
            )
            if minimum_distance > settings.MAX_FIRE_DISTANCE_MILES:
                continue

            properties = self._feature_properties(feature)
            incident_name = self._incident_name(properties) or "Incident"
            unique_id = self._incident_identifier(properties, latitude, longitude)
            if unique_id in seen_ids:
                continue
            seen_ids.add(unique_id)

            acres_burned = self._to_float(
                self._first_value(
                    properties,
                    "DailyAcres",
                    "GISAcres",
                    "Acres",
                    "poly_Acres",
                    "CalculatedAcres",
                    "acres",
                )
            )
            percent_contained = self._to_float(
                self._first_value(
                    properties,
                    "PercentContained",
                    "PERCENT_CONTAINED",
                    "percent_contained",
                    "Containment",
                    "per_contain",
                )
            )
            growth_acres_24h = self._to_float(
                self._first_value(
                    properties,
                    "DailyGrowth",
                    "DailyAcresChange",
                    "growth_acres_24h",
                    "Acres24Hr",
                )
            )
            observed_at = self._first_value(
                properties,
                "DateCurrent",
                "ModifiedOnDateTime",
                "LoadDate",
                "last_updated",
                "observed_at",
            )

            metadata: dict[str, Any] = {
                "intelligence_source": "Configured perimeter feed",
                "minimum_watch_distance_miles": round(minimum_distance, 1),
            }
            for source_key, target_key in (
                ("IRWINID", "irwin_id"),
                ("IRWIN_ID", "irwin_id"),
                ("UniqueId", "incident_unique_id"),
                ("UniqueID", "incident_unique_id"),
                ("IncidentTypeCategory", "incident_type"),
                ("POOState", "state"),
                ("POOCounty", "county"),
                ("FireDiscoveryDateTime", "discovered_at"),
            ):
                value = properties.get(source_key)
                if value not in (None, ""):
                    metadata[target_key] = value
            if growth_acres_24h is not None:
                metadata["growth_acres_24h"] = growth_acres_24h
            if observed_at not in (None, ""):
                metadata["observed_at"] = observed_at

            records.append(
                {
                    "Name": incident_name,
                    "UniqueId": unique_id,
                    "Latitude": latitude,
                    "Longitude": longitude,
                    "AcresBurned": acres_burned or 0.0,
                    "PercentContained": percent_contained or 0.0,
                    "Source": "PERIMETER",
                    "ObservedAt": observed_at,
                    "PolygonData": geometry if isinstance(geometry, dict) else None,
                    "ExtraMetadata": metadata,
                }
            )

        return records

    async def fetch_incident_perimeters(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any] | None:
        """Fetch nearest incident perimeter summary for a watch location."""
        features = await self.fetch_incident_perimeter_features()
        if not features:
            return None

        nearest = self._nearest_feature(features, latitude, longitude)
        if nearest is None:
            return {
                "source": "Configured perimeter feed",
                "incident_count": len(features),
                "nearest_incident": None,
            }

        properties = self._feature_properties(nearest["feature"])
        return {
            "source": "Configured perimeter feed",
            "incident_count": len(features),
            "nearest_incident": {
                "name": self._incident_name(properties) or "Incident",
                "acres": self._to_float(
                    self._first_value(
                        properties,
                        "DailyAcres",
                        "GISAcres",
                        "Acres",
                        "poly_Acres",
                        "CalculatedAcres",
                        "acres",
                    )
                ),
                "percent_contained": self._to_float(
                    self._first_value(
                        properties,
                        "PercentContained",
                        "PERCENT_CONTAINED",
                        "percent_contained",
                        "Containment",
                        "per_contain",
                    )
                ),
                "growth_acres_24h": self._to_float(
                    self._first_value(
                        properties,
                        "DailyGrowth",
                        "DailyAcresChange",
                        "growth_acres_24h",
                        "Acres24Hr",
                    )
                ),
                "distance_miles": nearest["distance_miles"],
                "observed_at": self._first_value(
                    properties,
                    "DateCurrent",
                    "ModifiedOnDateTime",
                    "LoadDate",
                    "last_updated",
                    "observed_at",
                ),
            },
        }

    async def fetch_road_closures(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any] | None:
        """Fetch nearby configured road-closure intelligence."""
        if not settings.ROAD_CLOSURES_GEOJSON_URL:
            return None

        features = await self._fetch_geojson_features(settings.ROAD_CLOSURES_GEOJSON_URL)
        if not features:
            return None

        nearby: list[dict[str, Any]] = []
        for feature in features:
            reference_point = self._feature_reference_point(feature)
            if reference_point is None:
                continue
            feature_lat, feature_lon = reference_point
            distance = self.haversine_distance(latitude, longitude, feature_lat, feature_lon)
            if distance > 30.0:
                continue
            nearby.append(
                {
                    "feature": feature,
                    "distance_miles": round(distance, 1),
                }
            )

        nearby.sort(key=lambda item: item["distance_miles"])
        nearest = nearby[0] if nearby else None
        nearest_feature = nearest["feature"] if nearest else None
        nearest_properties = self._feature_properties(nearest_feature)

        return {
            "source": "Configured road closure feed",
            "closure_count_nearby": len(nearby),
            "nearest_closure": (
                {
                    "distance_miles": nearest["distance_miles"],
                    "road_name": self._first_value(
                        nearest_properties, "road_name", "name", "RouteName"
                    ),
                    "status": self._first_value(
                        nearest_properties,
                        "status",
                        "closure_status",
                        "Status",
                    )
                    or "CLOSED",
                }
                if nearest is not None
                else None
            ),
        }

    async def fetch_lightning_nowcast(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any] | None:
        """Fetch nearby configured lightning nowcast intelligence."""
        if not settings.LIGHTNING_NOWCAST_URL:
            return None

        payload = await self._fetch_json_payload(settings.LIGHTNING_NOWCAST_URL)
        if payload is None:
            return None

        strikes = payload.get("strikes") if isinstance(payload, dict) else payload
        if not isinstance(strikes, list):
            return None

        nearby: list[dict[str, Any]] = []
        for strike in strikes:
            if not isinstance(strike, dict):
                continue
            strike_lat = self._to_float(strike.get("latitude"))
            strike_lon = self._to_float(strike.get("longitude"))
            if strike_lat is None or strike_lon is None:
                continue

            distance = self.haversine_distance(latitude, longitude, strike_lat, strike_lon)
            if distance > 40.0:
                continue
            nearby.append(
                {
                    "distance_miles": round(distance, 1),
                    "observed_at": strike.get("observed_at") or strike.get("time"),
                }
            )

        nearby.sort(key=lambda item: item["distance_miles"])
        return {
            "source": "Configured lightning nowcast feed",
            "strikes_nearby_count": len(nearby),
            "nearest_strike": nearby[0] if nearby else None,
            "elevated_ignition_risk": len(nearby) >= 3,
        }

    async def collect_for_location(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Collect configured external intelligence for one location."""
        incident_perimeters, road_closures, lightning_nowcast = await asyncio.gather(
            self.fetch_incident_perimeters(latitude, longitude),
            self.fetch_road_closures(latitude, longitude),
            self.fetch_lightning_nowcast(latitude, longitude),
        )

        feed_status = {
            "incident_perimeters": self._configured_feed_status(
                bool(settings.FIRE_PERIMETER_GEOJSON_URL),
                incident_perimeters,
            ),
            "road_closures": self._configured_feed_status(
                bool(settings.ROAD_CLOSURES_GEOJSON_URL),
                road_closures,
            ),
            "lightning_nowcast": self._configured_feed_status(
                bool(settings.LIGHTNING_NOWCAST_URL),
                lightning_nowcast,
            ),
        }

        return {
            "incident_perimeters": incident_perimeters,
            "road_closures": road_closures,
            "lightning_nowcast": lightning_nowcast,
            "feed_status": feed_status,
            "reliability": self._summarize_reliability(feed_status),
        }

    async def collect_for_fire_and_location(
        self,
        fire_latitude: float,
        fire_longitude: float,
        location_latitude: float,
        location_longitude: float,
    ) -> dict[str, Any]:
        """Collect perimeter context around the fire and ops signals around the watch location."""
        incident_perimeters, road_closures, lightning_nowcast = await asyncio.gather(
            self.fetch_incident_perimeters(fire_latitude, fire_longitude),
            self.fetch_road_closures(location_latitude, location_longitude),
            self.fetch_lightning_nowcast(location_latitude, location_longitude),
        )

        feed_status = {
            "incident_perimeters": self._configured_feed_status(
                bool(settings.FIRE_PERIMETER_GEOJSON_URL),
                incident_perimeters,
            ),
            "road_closures": self._configured_feed_status(
                bool(settings.ROAD_CLOSURES_GEOJSON_URL),
                road_closures,
            ),
            "lightning_nowcast": self._configured_feed_status(
                bool(settings.LIGHTNING_NOWCAST_URL),
                lightning_nowcast,
            ),
        }

        return {
            "incident_perimeters": incident_perimeters,
            "road_closures": road_closures,
            "lightning_nowcast": lightning_nowcast,
            "feed_status": feed_status,
            "reliability": self._summarize_reliability(feed_status),
        }

    @staticmethod
    def enrich_fire_context(
        fire_context: dict[str, Any],
        external_intelligence: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Apply perimeter-backed incident context to the current fire snapshot when nearby."""
        enriched = dict(fire_context)
        incident = ((external_intelligence or {}).get("incident_perimeters") or {}).get(
            "nearest_incident"
        ) or {}
        if not isinstance(incident, dict):
            return enriched

        incident_distance = ExternalIntelligenceService._to_float(incident.get("distance_miles"))
        if incident_distance is not None and incident_distance > 15.0:
            return enriched

        acres = ExternalIntelligenceService._to_float(incident.get("acres"))
        percent_contained = ExternalIntelligenceService._to_float(incident.get("percent_contained"))
        growth_acres_24h = ExternalIntelligenceService._to_float(incident.get("growth_acres_24h"))

        if acres is not None:
            enriched["acres_burned"] = acres
        if percent_contained is not None:
            enriched["percent_contained"] = percent_contained
        if growth_acres_24h is not None:
            enriched["growth_acres_24h"] = growth_acres_24h
        if incident.get("observed_at") not in (None, ""):
            enriched.setdefault("last_updated", incident.get("observed_at"))

        return enriched

    @staticmethod
    def summarize_threat_modifiers(
        external_intelligence: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Convert configured feed signals into threat-score modifiers and actions."""
        intelligence = external_intelligence or {}
        incident = (intelligence.get("incident_perimeters") or {}).get("nearest_incident") or {}
        road_closures = intelligence.get("road_closures") or {}
        lightning = intelligence.get("lightning_nowcast") or {}
        reliability = intelligence.get("reliability") or {}

        modifiers: list[str] = []
        recommendations: list[str] = []
        score_boost = 0.0

        growth_acres_24h = ExternalIntelligenceService._to_float(incident.get("growth_acres_24h"))
        if growth_acres_24h is not None and growth_acres_24h > 0.0:
            growth_boost = min(8.0, max(1.0, growth_acres_24h / 100.0))
            score_boost += growth_boost
            modifiers.append(
                f"Perimeter feed shows recent growth (+{growth_acres_24h:.0f} acres/24h)"
            )
            recommendations.append(
                "Perimeter feed shows recent growth; monitor operational updates closely"
            )

        closure_count = int(road_closures.get("closure_count_nearby") or 0)
        if closure_count > 0:
            score_boost += min(8.0, float(closure_count * 2))
            modifiers.append(f"Road closures reported nearby ({closure_count})")
            recommendations.append(
                "Road closures reported nearby; pre-plan alternate evacuation routes"
            )

        if lightning.get("elevated_ignition_risk"):
            score_boost += 4.0
            modifiers.append("Elevated nearby lightning activity may increase ignition risk")
            recommendations.append(
                "Elevated nearby lightning activity can increase new-start risk and complicate "
                "response"
            )

        success_ratio = ExternalIntelligenceService._to_float(reliability.get("success_ratio"))
        if success_ratio is not None and success_ratio < 1.0 and score_boost > 0.0:
            score_boost *= success_ratio

        return {
            "score_boost": round(score_boost, 1),
            "modifiers": modifiers,
            "recommendations": recommendations,
        }

    async def _fetch_geojson_features(self, url: str) -> list[dict[str, Any]]:
        """Fetch GeoJSON features from a configured feed."""
        payload = await self._fetch_json_payload(url)
        if not isinstance(payload, dict):
            return []

        features = payload.get("features")
        if not isinstance(features, list):
            return []
        return [feature for feature in features if isinstance(feature, dict)]

    async def _fetch_json_payload(self, url: str) -> Any:
        """Fetch JSON payload from a configured feed."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=settings.FIRE_DATA_TIMEOUT)
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            logger.warning("Configured intelligence feed fetch failed for %s: %s", url, exc)
            return None

    @classmethod
    def _nearest_feature(
        cls,
        features: list[dict[str, Any]],
        latitude: float,
        longitude: float,
    ) -> dict[str, Any] | None:
        """Return the nearest feature and distance to a location."""
        candidates: list[dict[str, Any]] = []
        for feature in features:
            reference_point = cls._feature_reference_point(feature)
            if reference_point is None:
                continue
            feature_lat, feature_lon = reference_point
            distance = cls.haversine_distance(latitude, longitude, feature_lat, feature_lon)
            candidates.append(
                {
                    "feature": feature,
                    "distance_miles": round(distance, 1),
                }
            )

        if not candidates:
            return None
        return min(candidates, key=lambda item: item["distance_miles"])

    @classmethod
    def _feature_reference_point(cls, feature: dict[str, Any]) -> tuple[float, float] | None:
        """Approximate a feature with a representative latitude/longitude."""
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(geometry, dict):
            return None
        return cls._geometry_reference_point(geometry)

    @classmethod
    def _geometry_reference_point(cls, geometry: dict[str, Any]) -> tuple[float, float] | None:
        """Approximate a geometry centroid using its coordinate vertices."""
        points = list(cls._iter_coordinate_points(geometry.get("coordinates")))
        if not points:
            return None

        avg_longitude = sum(point[0] for point in points) / len(points)
        avg_latitude = sum(point[1] for point in points) / len(points)
        return avg_latitude, avg_longitude

    @classmethod
    def _iter_coordinate_points(cls, coordinates: Any):
        """Yield GeoJSON coordinate pairs from arbitrarily nested geometry coordinates."""
        if not isinstance(coordinates, list) or not coordinates:
            return

        first = coordinates[0]
        if isinstance(first, (int, float)):
            if len(coordinates) >= 2:
                yield float(coordinates[0]), float(coordinates[1])
            return

        for child in coordinates:
            yield from cls._iter_coordinate_points(child)

    @staticmethod
    def _incident_name(properties: dict[str, Any]) -> str | None:
        """Return the most informative incident name from perimeter properties."""
        value = ExternalIntelligenceService._first_value(
            properties,
            "IncidentName",
            "Incident_Name",
            "incident_name",
            "FireName",
            "FIRE_NAME",
            "name",
            "IRWINID",
        )
        return str(value).strip() if value not in (None, "") else None

    @classmethod
    def _incident_identifier(
        cls,
        properties: dict[str, Any],
        latitude: float,
        longitude: float,
    ) -> str:
        """Build a stable perimeter-backed incident identifier."""
        source_id = cls._first_value(
            properties,
            "IRWINID",
            "IRWIN_ID",
            "UniqueId",
            "UniqueID",
            "GlobalID",
            "OBJECTID",
            "OBJECTID_1",
        )
        if source_id not in (None, ""):
            return f"perimeter:{source_id}"

        incident_name = cls._incident_name(properties) or "incident"
        normalized_name = re.sub(r"[^a-z0-9]+", "-", incident_name.lower()).strip("-") or "incident"
        return f"perimeter:{normalized_name}:{latitude:.4f}:{longitude:.4f}"

    @staticmethod
    def _feature_properties(feature: object) -> dict[str, Any]:
        """A GeoJSON feature's properties; GeoJSON allows null, which reads as none."""
        properties = feature.get("properties") if isinstance(feature, dict) else None
        return properties if isinstance(properties, dict) else {}

    @staticmethod
    def _first_value(mapping: dict[str, Any], *keys: str) -> Any:
        """Return the first populated value from a mapping."""
        for key in keys:
            value = mapping.get(key)
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Convert a value to float when possible."""
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _configured_feed_status(is_configured: bool, payload: Any) -> str:
        """Return configured feed health status."""
        if not is_configured:
            return "not_configured"
        return "ok" if payload is not None else "degraded"

    @staticmethod
    def _summarize_reliability(feed_status: dict[str, str]) -> dict[str, Any]:
        """Summarize configured feed reliability similar to the edge overlay."""
        statuses = list(feed_status.values())
        expected = [status for status in statuses if status != "not_configured"]
        ok_count = sum(1 for status in expected if status == "ok")
        degraded_count = sum(1 for status in expected if status == "degraded")
        expected_count = len(expected)
        success_ratio = ok_count / expected_count if expected_count else 1.0

        overall_status = "ok"
        if expected_count == 0:
            overall_status = "not_configured"
        elif success_ratio < 0.6:
            overall_status = "down"
        elif success_ratio < 0.9:
            overall_status = "degraded"

        return {
            "status": overall_status,
            "ok_count": ok_count,
            "degraded_count": degraded_count,
            "not_configured_count": sum(1 for status in statuses if status == "not_configured"),
            "expected_count": expected_count,
            "success_ratio": round(success_ratio, 3),
            "slo_target_success_ratio": 0.9,
            "slo_met": success_ratio >= 0.9 if expected_count else None,
        }

    @staticmethod
    def haversine_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate great-circle distance in miles between two points."""
        radius_miles = 3959.87433
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
            math.radians,
            [lat1, lon1, lat2, lon2],
        )
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad
        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        return radius_miles * c
