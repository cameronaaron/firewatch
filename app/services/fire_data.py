"""Fire data fetching and processing service."""

import asyncio
import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import FireDataFetchError
from app.models.database import Fire, Location
from app.services.external_intelligence import ExternalIntelligenceService
from app.services.satellite_service import FireDetection, SatelliteFireService

logger = logging.getLogger(__name__)
settings = get_settings()


class FireDataService:
    """Service for fetching and processing fire data."""

    # In-memory cache for historical data
    _distance_history_cache: dict[str, list[dict[str, Any]]] = {}
    _satellite_service: SatelliteFireService | None = None
    _external_intelligence_service: ExternalIntelligenceService | None = None

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance in miles between two points.

        Returns:
            Distance in miles
        """
        R = 3959.87433  # Earth's radius in miles
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate bearing from point 1 to point 2 in degrees [0, 360).

        Returns:
            Bearing in degrees
        """
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1
        x = math.cos(lat2) * math.sin(dlon)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.degrees(math.atan2(x, y))

        return (bearing + 360) % 360

    @staticmethod
    def get_cardinal_direction(bearing: float) -> str:
        """Convert bearing to cardinal direction."""
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = round(bearing / 45) % 8
        return directions[index]

    @classmethod
    def _get_satellite_service(cls) -> SatelliteFireService:
        """Create the satellite service lazily so tests can replace it cheaply."""
        if cls._satellite_service is None:
            cls._satellite_service = SatelliteFireService()
        return cls._satellite_service

    @classmethod
    def _get_external_intelligence_service(cls) -> ExternalIntelligenceService:
        """Create the external intelligence service lazily for configurable feeds."""
        if cls._external_intelligence_service is None:
            cls._external_intelligence_service = ExternalIntelligenceService()
        return cls._external_intelligence_service

    @classmethod
    async def fetch_fires_from_api(cls) -> list[dict[str, Any]]:
        """
        Fetch fire data from the best available wildfire feeds.

        Sources:
        - CAL FIRE incidents API for authoritative California incident metadata.
        - NASA EONET wildfire events for wider geographic coverage and redundancy.

        Raises:
            FireDataFetchError: If API fetch fails
        """
        source_records: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}
        failures: list[str] = []

        try:
            async with httpx.AsyncClient() as client:
                for source_name, fetcher in (
                    ("CAL_FIRE", cls._fetch_calfire_fires),
                    ("EONET", cls._fetch_eonet_fires),
                ):
                    try:
                        records = await fetcher(client)
                        if records:
                            source_records.extend(records)
                            source_counts[source_name] = len(records)
                    except Exception as exc:
                        failures.append(f"{source_name}: {exc}")
                        logger.warning("%s fire fetch failed: %s", source_name, exc)

            if not source_records:
                message = failures[0] if failures else "No fire data sources returned usable data"
                raise FireDataFetchError(f"Failed to fetch fire data: {message}")

            merged_records = cls._merge_source_records(source_records)
            logger.info(
                "Fetched %s raw fire records across %s and merged into %s incidents",
                len(source_records),
                source_counts,
                len(merged_records),
            )
            return merged_records

        except FireDataFetchError:
            raise
        except httpx.HTTPError as e:
            logger.error("HTTP error fetching fire data: %s", e)
            raise FireDataFetchError(f"Failed to fetch fire data: {str(e)}") from e
        except Exception as e:
            logger.error("Unexpected error fetching fire data: %s", e)
            raise FireDataFetchError(f"Failed to fetch fire data: {str(e)}") from e

    @staticmethod
    async def _fetch_calfire_fires(client: httpx.AsyncClient) -> list[dict[str, Any]]:
        """Fetch active incidents from CAL FIRE."""
        response = await client.get(
            f"{settings.CALFIRE_API_URL}?inactive=false",
            timeout=settings.FIRE_DATA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list):
            raise ValueError("Expected list response from CAL FIRE API")

        normalized: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            record = dict(item)
            record.setdefault("Source", "CAL_FIRE")
            normalized.append(record)

        return normalized

    @classmethod
    async def _fetch_eonet_fires(cls, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        """Fetch open wildfire events from NASA EONET."""
        response = await client.get(
            "https://eonet.gsfc.nasa.gov/api/v3/events",
            params={
                "status": "open",
                "category": "wildfires",
                "limit": 500,
            },
            timeout=settings.FIRE_DATA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("Expected mapping response from EONET API")

        events = data.get("events", [])
        if not isinstance(events, list):
            raise ValueError("Expected events list from EONET API")

        normalized: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                continue

            geometry = cls._latest_point_geometry(event.get("geometry", []))
            if not geometry:
                continue

            coordinates = geometry.get("coordinates")
            if not isinstance(coordinates, list) or len(coordinates) < 2:
                continue

            longitude, latitude = coordinates[0], coordinates[1]
            magnitude_value = geometry.get("magnitudeValue")
            magnitude_unit = str(geometry.get("magnitudeUnit") or "").lower()

            acres_burned = 0.0
            if magnitude_value is not None and magnitude_unit == "acres":
                acres_burned = float(magnitude_value)

            normalized.append(
                {
                    "Name": event.get("title") or "Unknown",
                    "UniqueId": event.get("id") or event.get("title") or "Unknown",
                    "Latitude": latitude,
                    "Longitude": longitude,
                    "AcresBurned": acres_burned,
                    "PercentContained": 0.0,
                    "Source": "EONET",
                    "Sources": [
                        source.get("id")
                        for source in event.get("sources", [])
                        if isinstance(source, dict) and source.get("id")
                    ],
                    "ObservedAt": geometry.get("date"),
                }
            )

        return normalized

    @staticmethod
    def _latest_point_geometry(geometries: Any) -> dict[str, Any] | None:
        """Return the most recent point geometry from an EONET event."""
        if not isinstance(geometries, list):
            return None

        point_geometries = [
            geometry
            for geometry in geometries
            if isinstance(geometry, dict) and geometry.get("type") == "Point"
        ]
        if not point_geometries:
            return None

        return max(point_geometries, key=lambda geometry: str(geometry.get("date") or ""))

    @classmethod
    def _merge_source_records(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge duplicate incidents across wildfire data sources."""
        merged: dict[str, dict[str, Any]] = {}

        for record in records:
            key = cls._record_merge_key(record)
            merge_key = key
            if merge_key not in merged:
                merge_key = cls._find_spatial_merge_key(record, merged) or key

            if merge_key not in merged:
                merged[key] = dict(record)
                continue

            merged[merge_key] = cls._combine_source_records(merged[merge_key], record)

        return list(merged.values())

    @staticmethod
    def _record_merge_key(record: dict[str, Any]) -> str:
        """Build a source-independent dedupe key for a fire record."""
        name = FireDataService._normalized_incident_name(record.get("Name")) or "unknown"

        try:
            latitude = round(float(record.get("Latitude") or 0.0), 2)
            longitude = round(float(record.get("Longitude") or 0.0), 2)
        except (TypeError, ValueError):
            latitude = 0.0
            longitude = 0.0

        return f"{name}:{latitude}:{longitude}"

    @classmethod
    def _find_spatial_merge_key(
        cls,
        record: dict[str, Any],
        merged: dict[str, dict[str, Any]],
    ) -> str | None:
        """Merge FIRMS hotspots into the nearest known incident when they align spatially."""
        record_family = cls._source_family(record.get("Source"))
        if record_family not in {"FIRMS", "PERIMETER"}:
            return None

        try:
            record_lat = float(record.get("Latitude") or 0.0)
            record_lon = float(record.get("Longitude") or 0.0)
        except (TypeError, ValueError):
            return None

        best_key: str | None = None
        best_score: tuple[int, int, float] | None = None
        record_name = cls._normalized_incident_name(record.get("Name"))
        for key, existing in merged.items():
            try:
                existing_lat = float(existing.get("Latitude") or 0.0)
                existing_lon = float(existing.get("Longitude") or 0.0)
            except (TypeError, ValueError):
                continue

            existing_family = cls._source_family(existing.get("Source"))
            distance_miles = cls.haversine_distance(
                record_lat, record_lon, existing_lat, existing_lon
            )
            existing_name = cls._normalized_incident_name(existing.get("Name"))
            same_incident_name = bool(
                record_name and existing_name and record_name == existing_name
            )

            if record_family == "PERIMETER":
                threshold_miles = 12.0 if same_incident_name else 5.0
                if distance_miles > threshold_miles:
                    continue
                score = (
                    0 if same_incident_name else 1,
                    0 if existing_family != "PERIMETER" else 1,
                    distance_miles,
                )
            else:
                threshold_miles = 1.5 if existing_family == "FIRMS" else 3.0
                if distance_miles > threshold_miles:
                    continue
                score = (
                    0 if existing_family != "FIRMS" else 1,
                    0,
                    distance_miles,
                )

            if best_score is None or score < best_score:
                best_key = key
                best_score = score

        return best_key

    @staticmethod
    def _normalized_incident_name(value: Any) -> str:
        """Normalize incident names for source-agnostic merge comparisons."""
        return " ".join(str(value or "").lower().split())

    @staticmethod
    def _source_family(source: Any) -> str:
        """Normalize a source label to a merge priority family."""
        normalized = str(source or "").strip().upper()
        if normalized.startswith("PERIMETER"):
            return "PERIMETER"
        if normalized.startswith("CAL_FIRE"):
            return "CAL_FIRE"
        if normalized.startswith("FIRMS"):
            return "FIRMS"
        if normalized.startswith("EONET"):
            return "EONET"
        return normalized

    @classmethod
    def _combine_source_records(
        cls,
        primary: dict[str, Any],
        secondary: dict[str, Any],
    ) -> dict[str, Any]:
        """Prefer richer source metadata while preserving source provenance."""
        merged = dict(primary)
        source_priority = {"PERIMETER": 5, "CAL_FIRE": 4, "FIRMS": 3, "EONET": 2}
        primary_source = cls._source_family(primary.get("Source"))
        secondary_source = cls._source_family(secondary.get("Source"))

        if source_priority.get(secondary_source, 0) > source_priority.get(primary_source, 0):
            merged = dict(secondary)

        for field in ("Name", "UniqueId", "Latitude", "Longitude"):
            if merged.get(field) in (None, "", 0, 0.0):
                merged[field] = secondary.get(field)

        if (
            float(merged.get("AcresBurned") or 0.0) == 0.0
            and float(secondary.get("AcresBurned") or 0.0) > 0.0
        ):
            merged["AcresBurned"] = secondary.get("AcresBurned")

        if (
            float(merged.get("PercentContained") or 0.0) == 0.0
            and float(secondary.get("PercentContained") or 0.0) > 0.0
        ):
            merged["PercentContained"] = secondary.get("PercentContained")

        for field in ("ObservedAt", "Confidence", "RadiativePower", "BrightnessTemp"):
            if merged.get(field) in (None, "", 0, 0.0) and secondary.get(field) not in (
                None,
                "",
                0,
                0.0,
            ):
                merged[field] = secondary.get(field)

        if merged.get("PolygonData") is None and secondary.get("PolygonData") is not None:
            merged["PolygonData"] = secondary.get("PolygonData")

        extra_metadata: dict[str, Any] = {}
        if isinstance(primary.get("ExtraMetadata"), dict):
            extra_metadata.update(primary["ExtraMetadata"])
        if isinstance(secondary.get("ExtraMetadata"), dict):
            extra_metadata.update(secondary["ExtraMetadata"])
        if extra_metadata:
            merged["ExtraMetadata"] = extra_metadata

        sources = {
            str(primary.get("Source") or "").strip(),
            str(secondary.get("Source") or "").strip(),
        }
        merged["Source"] = "+".join(sorted(source for source in sources if source))
        return merged

    @classmethod
    async def _fetch_configured_perimeter_fires(
        cls,
        locations: list[Location],
    ) -> list[dict[str, Any]]:
        """Fetch configured perimeter features and normalize them into fire records."""
        try:
            intelligence_service = cls._get_external_intelligence_service()
            records = await intelligence_service.build_perimeter_fire_records(locations)
            if records:
                logger.info("Fetched %s configured perimeter-backed incidents", len(records))
            return records
        except Exception as exc:
            logger.warning("Configured perimeter fire fetch failed: %s", exc)
            return []

    @classmethod
    async def _fetch_firms_near_locations(cls, locations: list[Location]) -> list[dict[str, Any]]:
        """Fetch and normalize FIRMS hotspots near monitored locations."""
        coordinate_queries: list[tuple[float, float]] = []
        seen_queries: set[tuple[float, float]] = set()
        for location in locations:
            if location.latitude is None or location.longitude is None:
                continue

            try:
                latitude = float(location.latitude)
                longitude = float(location.longitude)
            except (TypeError, ValueError):
                continue

            query_key = (round(latitude, 4), round(longitude, 4))
            if query_key in seen_queries:
                continue
            seen_queries.add(query_key)
            coordinate_queries.append((latitude, longitude))

        if not coordinate_queries:
            return []

        satellite_service = cls._get_satellite_service()
        results = await asyncio.gather(
            *[
                satellite_service.get_firms_detections(latitude, longitude)
                for latitude, longitude in coordinate_queries
            ],
            return_exceptions=True,
        )

        detections: list[FireDetection] = []
        for (latitude, longitude), result in zip(coordinate_queries, results, strict=False):
            if isinstance(result, BaseException):
                logger.warning(
                    "FIRMS hotspot fetch failed near (%s, %s): %s", latitude, longitude, result
                )
                continue
            detections.extend(result)

        if not detections:
            return []

        normalized = [cls._normalize_firms_detection(detection) for detection in detections]
        merged = cls._merge_source_records(normalized)
        logger.info(
            "Fetched %s FIRMS detections near monitored locations and normalized them into %s "
            "hotspots",
            len(detections),
            len(merged),
        )
        return merged

    @staticmethod
    def _normalize_firms_detection(detection: FireDetection) -> dict[str, Any]:
        """Convert a FIRMS hotspot detection to the normalized fire record shape."""
        sensor = str(detection.sensor or "FIRMS").strip().upper() or "FIRMS"
        timestamp = detection.timestamp.astimezone(UTC)
        timestamp_token = timestamp.strftime("%Y%m%dT%H%M%S")
        return {
            "Name": f"{sensor} satellite hotspot",
            "UniqueId": (
                f"firms-{sensor.lower()}-{timestamp_token}"
                f"-{detection.latitude:.4f}-{detection.longitude:.4f}"
            ),
            "Latitude": detection.latitude,
            "Longitude": detection.longitude,
            "AcresBurned": 0.0,
            "PercentContained": 0.0,
            "Source": f"FIRMS:{sensor}",
            "ObservedAt": timestamp.isoformat(),
            "Confidence": detection.confidence,
            "RadiativePower": detection.radiative_power,
            "BrightnessTemp": detection.brightness_temp,
        }

    @classmethod
    async def process_fire_data(
        cls, db: Session, locations: list[Location]
    ) -> list[dict[str, Any]]:
        """Fetch fires and compute distances to all monitored locations."""
        try:
            raw_fires: list[dict[str, Any]] = []
            fetch_error: FireDataFetchError | None = None
            try:
                raw_fires = await cls.fetch_fires_from_api()
            except FireDataFetchError as exc:
                fetch_error = exc
                logger.warning("Primary fire feeds failed; attempting FIRMS fallback: %s", exc)

            perimeter_fires = await cls._fetch_configured_perimeter_fires(locations)
            if perimeter_fires:
                raw_fires = cls._merge_source_records(raw_fires + perimeter_fires)

            firms_fires = await cls._fetch_firms_near_locations(locations)
            if firms_fires:
                raw_fires = cls._merge_source_records(raw_fires + firms_fires)

            if not raw_fires and fetch_error is not None:
                raise fetch_error

            processed_fires = []
            seen_keys = set()

            for fire_data in raw_fires:
                try:
                    # Parse fire coordinates
                    lat = float(fire_data.get("Latitude") or 0)
                    lon = float(fire_data.get("Longitude") or 0)
                    fire_id = fire_data.get("UniqueId") or fire_data.get("Name", "Unknown")

                    if not fire_id or lat == 0 or lon == 0:
                        continue

                    # Avoid duplicates
                    key = f"{fire_id}_{lat}_{lon}"
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    # Calculate distances to each location
                    distances: dict[str, float] = {}
                    directions: dict[str, str] = {}
                    bearings: dict[str, float] = {}

                    for location in locations:
                        if location.latitude is None or location.longitude is None:
                            continue

                        distance = cls.haversine_distance(
                            lat, lon, location.latitude, location.longitude
                        )

                        if distance <= settings.MAX_FIRE_DISTANCE_MILES:
                            bearing = cls.calculate_bearing(
                                lat, lon, location.latitude, location.longitude
                            )
                            cardinal = cls.get_cardinal_direction(bearing)

                            distances[location.location_id] = round(distance, 1)
                            directions[location.location_id] = f"{cardinal} ({int(bearing)}°)"
                            bearings[location.location_id] = bearing

                    if not distances:
                        continue

                    # Extract fire metrics
                    acres_burned = float(fire_data.get("AcresBurned") or 0)
                    percent_contained = float(fire_data.get("PercentContained") or 0)

                    fire_record = {
                        "fire_id": fire_id,
                        "name": fire_data.get("Name", "Unknown"),
                        "latitude": round(lat, 4),
                        "longitude": round(lon, 4),
                        "acres_burned": acres_burned,
                        "percent_contained": percent_contained,
                        "polygon_data": fire_data.get("PolygonData")
                        if isinstance(fire_data.get("PolygonData"), dict)
                        else None,
                        "extra_metadata": cls._build_fire_metadata(fire_data),
                        "distances": distances,
                        "directions": directions,
                        "bearings": bearings,
                        "movement_data": {},
                    }

                    # Calculate movement data for each location
                    for location_id in distances:
                        moving_towards, speed = cls.calculate_movement(
                            fire_id, location_id, distances[location_id]
                        )
                        fire_record["movement_data"][location_id] = {
                            "moving_towards": moving_towards,
                            "speed": speed,
                        }

                    processed_fires.append(fire_record)

                except Exception as e:
                    logger.warning(f"Error processing fire record: {e}")
                    continue

            logger.info(f"Processed {len(processed_fires)} nearby fires")
            return processed_fires

        except FireDataFetchError:
            raise
        except Exception as e:
            logger.error(f"Error processing fire data: {e}")
            raise FireDataFetchError(f"Failed to process fire data: {str(e)}") from e

    @staticmethod
    def _build_fire_metadata(fire_data: dict[str, Any]) -> dict[str, Any]:
        """Build persisted fire metadata from merged source records."""
        metadata = dict(fire_data.get("ExtraMetadata") or {})
        source_value = str(fire_data.get("Source") or "").strip()
        if source_value:
            metadata.setdefault(
                "merged_sources",
                [source for source in source_value.split("+") if source],
            )

        observed_at = fire_data.get("ObservedAt")
        if observed_at not in (None, ""):
            metadata.setdefault("observed_at", observed_at)

        return metadata

    @staticmethod
    def upsert_fire(db: Session, fire_data: dict[str, Any]) -> Fire:
        """Insert or refresh a fire from a fetch; being reported makes it active again."""
        db_fire: Fire | None = db.query(Fire).filter(Fire.fire_id == fire_data["fire_id"]).first()
        if db_fire is None:
            db_fire = Fire(
                fire_id=fire_data["fire_id"],
                name=fire_data["name"],
                latitude=fire_data["latitude"],
                longitude=fire_data["longitude"],
                acres_burned=fire_data["acres_burned"],
                percent_contained=fire_data["percent_contained"],
                polygon_data=fire_data.get("polygon_data"),
                extra_metadata=fire_data.get("extra_metadata") or {},
            )
            db.add(db_fire)
        else:
            db_fire.name = fire_data["name"]
            db_fire.latitude = fire_data["latitude"]
            db_fire.longitude = fire_data["longitude"]
            db_fire.acres_burned = fire_data["acres_burned"]
            db_fire.percent_contained = fire_data["percent_contained"]
            db_fire.polygon_data = fire_data.get("polygon_data") or db_fire.polygon_data
            db_fire.extra_metadata = {
                **(db_fire.extra_metadata or {}),
                **(fire_data.get("extra_metadata") or {}),
            }
            db_fire.is_active = True
            db_fire.last_updated = datetime.now(UTC)
        db.flush()
        return db_fire

    @staticmethod
    def retire_stale_fires(db: Session, now: datetime) -> int:
        """Deactivate fires no successful fetch has reported for FIRE_STALE_HOURS.

        Call only after a successful fetch: a failed one says nothing about
        whether a fire is out. The delay lets short partial outages pass.
        """
        cutoff = now - timedelta(hours=settings.FIRE_STALE_HOURS)
        stale = db.query(Fire).filter(Fire.is_active, Fire.last_updated < cutoff).all()
        for fire in stale:
            fire.is_active = False
        return len(stale)

    @classmethod
    def calculate_movement(
        cls, fire_id: str, location_id: str, current_distance: float
    ) -> tuple[bool, float]:
        """
        Determine if fire is moving towards location and calculate speed.

        Args:
            fire_id: Fire identifier
            location_id: Location identifier
            current_distance: Current distance in miles

        Returns:
            Tuple of (moving_towards: bool, speed_mph: float)
        """
        cache_key = f"{fire_id}_{location_id}"

        if cache_key not in cls._distance_history_cache:
            cls._distance_history_cache[cache_key] = []

        history = cls._distance_history_cache[cache_key]

        # Add current measurement
        history.append({"timestamp": datetime.now(UTC), "distance": current_distance})

        cutoff = datetime.now(UTC) - timedelta(hours=settings.HISTORICAL_DATA_RETENTION_HOURS)
        cls._distance_history_cache[cache_key] = [h for h in history if h["timestamp"] > cutoff]

        # Calculate movement if we have at least 2 data points
        if len(cls._distance_history_cache[cache_key]) < 2:
            return False, 0.0

        history = cls._distance_history_cache[cache_key]
        prev = history[-2]
        curr = history[-1]

        time_diff_hours = (curr["timestamp"] - prev["timestamp"]).total_seconds() / 3600.0
        if time_diff_hours <= 0:
            return False, 0.0

        distance_diff = prev["distance"] - current_distance
        speed = abs(distance_diff) / time_diff_hours
        moving_towards = current_distance < prev["distance"]

        return moving_towards, round(speed, 1)

    @classmethod
    def get_alert_level(
        cls, distance: float | None, acres_burned: float = 0.0, percent_contained: float = 0.0
    ) -> str:
        """
        Determine alert level based on distance, size, and containment.

        Args:
            distance: Distance to fire in miles
            acres_burned: Acres burned by fire
            percent_contained: Percentage of fire contained
        """
        if distance is None:
            return "UNKNOWN"

        adjusted_dist = distance

        # Adjust for large fire size
        if acres_burned > 5000:
            adjusted_dist *= 0.5
        elif acres_burned > 1000:
            adjusted_dist *= 0.7

        # Adjust for low containment
        if percent_contained < 30:
            adjusted_dist *= 0.7
        if percent_contained < 10:
            adjusted_dist *= 0.5

        # Compare with thresholds
        if adjusted_dist <= settings.EVACUATION_DISTANCE_EVACUATE:
            return "EVACUATE"
        elif adjusted_dist <= settings.EVACUATION_DISTANCE_WARNING:
            return "WARNING"
        elif adjusted_dist <= settings.EVACUATION_DISTANCE_PREPARE:
            return "PREPARE"
        elif adjusted_dist <= settings.EVACUATION_DISTANCE_MONITOR:
            return "MONITOR"
        else:
            return "SAFE"
