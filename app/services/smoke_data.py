"""NOAA HMS smoke polygon service for location-based smoke assessment."""

from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import nearest_points, unary_union

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_KML_NAMESPACES = {"kml": "http://www.opengis.net/kml/2.2"}
_DESCRIPTION_PATTERN = re.compile(
    r"Start Time:\s*(?P<start>\d{7}\s+\d{4})UTC"
    r".*?End Time:\s*(?P<end>\d{7}\s+\d{4})UTC"
    r".*?Density:\s*(?P<density>[A-Za-z]+)"
    r".*?Satellite:\s*(?P<satellite>[A-Za-z\-0-9_]+)",
    re.IGNORECASE | re.DOTALL,
)
_DENSITY_RANK = {"LIGHT": 1, "MEDIUM": 2, "HEAVY": 3}


@dataclass(frozen=True)
class SmokePlume:
    """Parsed NOAA HMS smoke plume."""

    density: str
    geometry: Polygon | MultiPolygon
    start_time: datetime | None
    end_time: datetime | None
    satellite: str | None
    source: str = "NOAA_HMS"


class SmokeDataService:
    """Assess smoke exposure at a location using NOAA HMS daily smoke polygons."""

    _document_cache: dict[str, tuple[datetime, list[SmokePlume]]] = {}

    def __init__(self) -> None:
        self.base_url = settings.HMS_SMOKE_KML_BASE_URL.rstrip("/")
        self.timeout = settings.HMS_SMOKE_TIMEOUT
        self.lookback_days = settings.HMS_SMOKE_LOOKBACK_DAYS
        self.max_distance_miles = settings.HMS_SMOKE_MAX_DISTANCE_MILES

    def get_smoke_assessment(
        self,
        latitude: float,
        longitude: float,
        *,
        lookback_days: int | None = None,
        max_distance_miles: float | None = None,
    ) -> dict[str, Any] | None:
        """Return the strongest recent smoke plume affecting or near a location."""
        days = lookback_days or self.lookback_days
        distance_limit = max_distance_miles or self.max_distance_miles
        point = Point(float(longitude), float(latitude))

        best: dict[str, Any] | None = None
        for day_offset in range(days):
            target_date = datetime.now(UTC).date() - timedelta(days=day_offset)
            for plume in self._load_plumes_for_date(target_date):
                distance_miles = self._distance_to_geometry_miles(point, plume.geometry)
                if distance_miles > distance_limit:
                    continue

                candidate = {
                    "smoke_present": True,
                    "smoke_density": plume.density.title(),
                    "smoke_distance_miles": round(distance_miles, 1),
                    "smoke_scope": "overhead" if distance_miles == 0.0 else "nearby",
                    "smoke_source": plume.source,
                    "smoke_satellite": plume.satellite,
                    "smoke_observed_at": plume.end_time,
                    "smoke_started_at": plume.start_time,
                }
                if best is None or self._smoke_sort_key(candidate) < self._smoke_sort_key(best):
                    best = candidate

        return best

    def _load_plumes_for_date(self, target_date: date) -> list[SmokePlume]:
        """Load one day's smoke polygons, using a short in-memory cache."""
        cache_key = target_date.isoformat()
        cached = self._document_cache.get(cache_key)
        if cached and (datetime.now(UTC) - cached[0]).total_seconds() < settings.CACHE_TTL_SECONDS:
            return cached[1]

        url = self._build_daily_url(target_date)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                if response.status_code == 404:
                    return []
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("HMS smoke fetch failed for %s: %s", target_date, exc)
            return []
        except httpx.HTTPError as exc:
            logger.warning("HMS smoke fetch failed for %s: %s", target_date, exc)
            return []

        plumes = self._parse_kml(response.text)
        self._document_cache[cache_key] = (datetime.now(UTC), plumes)
        return plumes

    def _build_daily_url(self, target_date: date) -> str:
        """Build the HMS daily KML URL for a calendar date."""
        stamp = target_date.strftime("%Y%m%d")
        return f"{self.base_url}/{target_date.year}/{target_date.month:02d}/hms_smoke{stamp}.kml"

    def _parse_kml(self, kml_text: str) -> list[SmokePlume]:
        """Parse NOAA HMS KML into smoke plume objects."""
        root = ET.fromstring(kml_text)
        plumes: list[SmokePlume] = []
        for placemark in root.findall(".//kml:Placemark", _KML_NAMESPACES):
            style_url = (
                placemark.findtext("kml:styleUrl", default="", namespaces=_KML_NAMESPACES) or ""
            ).strip()
            if "Smoke_" not in style_url:
                continue

            description = (
                placemark.findtext("kml:description", default="", namespaces=_KML_NAMESPACES) or ""
            )
            density, start_time, end_time, satellite = self._parse_description(
                description, style_url
            )
            geometry = self._parse_geometry(placemark)
            if geometry is None:
                continue

            plumes.append(
                SmokePlume(
                    density=density,
                    geometry=geometry,
                    start_time=start_time,
                    end_time=end_time,
                    satellite=satellite,
                )
            )

        return plumes

    def _parse_description(
        self,
        description: str,
        style_url: str,
    ) -> tuple[str, datetime | None, datetime | None, str | None]:
        """Extract smoke metadata from the placemark description."""
        match = _DESCRIPTION_PATTERN.search(description)
        density = self._density_from_style(style_url)
        if not match:
            return density, None, None, None

        density = match.group("density").upper()
        satellite = match.group("satellite")
        start_time = self._parse_hms_timestamp(match.group("start"))
        end_time = self._parse_hms_timestamp(match.group("end"))
        return density, start_time, end_time, satellite

    @staticmethod
    def _parse_hms_timestamp(value: str) -> datetime | None:
        """Parse HMS timestamps like 2026140 1200 into UTC datetimes."""
        try:
            return datetime.strptime(value.strip(), "%Y%j %H%M").replace(tzinfo=UTC)
        except ValueError:
            return None

    @staticmethod
    def _density_from_style(style_url: str) -> str:
        """Infer smoke density from the KML style URL."""
        normalized = style_url.upper()
        if "SMOKE_HEAVY" in normalized:
            return "HEAVY"
        if "SMOKE_MEDIUM" in normalized:
            return "MEDIUM"
        return "LIGHT"

    def _parse_geometry(self, placemark: ET.Element) -> Polygon | MultiPolygon | None:
        """Parse polygon or multi-polygon geometry from a placemark."""
        polygons: list[Polygon] = []
        for coordinates_node in placemark.findall(
            ".//kml:Polygon//kml:coordinates", _KML_NAMESPACES
        ):
            polygon = self._polygon_from_coordinates(coordinates_node.text or "")
            if polygon is not None:
                polygons.append(polygon)

        if not polygons:
            return None
        if len(polygons) == 1:
            return polygons[0]
        return unary_union(polygons)

    @staticmethod
    def _polygon_from_coordinates(raw_coordinates: str) -> Polygon | None:
        """Build one shapely polygon from a KML coordinate list."""
        points = []
        for line in raw_coordinates.split():
            values = [segment.strip() for segment in line.split(",")]
            if len(values) < 2:
                continue
            try:
                lon = float(values[0])
                lat = float(values[1])
            except ValueError:
                continue
            points.append((lon, lat))

        if len(points) < 3:
            return None

        polygon = Polygon(points)
        if polygon.is_empty:
            return None
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            return None
        return polygon

    def _distance_to_geometry_miles(self, point: Point, geometry: Polygon | MultiPolygon) -> float:
        """Return the location-to-plume distance in miles."""
        if geometry.covers(point):
            return 0.0

        nearest_geometry_point = nearest_points(point, geometry)[1]
        return self._haversine_distance(
            point.y,
            point.x,
            nearest_geometry_point.y,
            nearest_geometry_point.x,
        )

    @staticmethod
    def _smoke_sort_key(candidate: dict[str, Any]) -> tuple[int, int, float, float]:
        """Prioritize overlapping, denser, newer, and nearer smoke plumes."""
        observed = candidate.get("smoke_observed_at")
        observed_ts = observed.timestamp() if isinstance(observed, datetime) else 0.0
        distance = float(candidate.get("smoke_distance_miles") or 0.0)
        density = str(candidate.get("smoke_density") or "LIGHT").upper()
        return (
            0 if distance == 0.0 else 1,
            -_DENSITY_RANK.get(density, 1),
            -observed_ts,
            distance,
        )

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in miles between two points."""
        radius_miles = 3959.87433
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return radius_miles * c
