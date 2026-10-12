"""Open-Meteo air-quality service for location-based exposure assessment."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AirQualityService:
    """Fetch AQ from Open-Meteo and optionally AirNow, then merge by confidence."""

    def __init__(self) -> None:
        self.base_url = settings.AIR_QUALITY_BASE_URL.rstrip("/")
        self.timeout = settings.AIR_QUALITY_TIMEOUT
        self.airnow_api_key = settings.AIRNOW_API_KEY
        self.airnow_base_url = settings.AIRNOW_BASE_URL.rstrip("/")
        self.airnow_radius_miles = settings.AIRNOW_RADIUS_MILES

    def get_air_quality_assessment(
        self, latitude: float, longitude: float
    ) -> dict[str, Any] | None:
        """Return merged air-quality conditions for a location when available."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                openmeteo = self._fetch_openmeteo(client, latitude, longitude)
                airnow = self._fetch_airnow(client, latitude, longitude)
        except httpx.HTTPError as exc:
            logger.warning("Air-quality fetch failed near (%s, %s): %s", latitude, longitude, exc)
            return None

        candidates = [source for source in (openmeteo, airnow) if source is not None]
        if not candidates:
            return None

        merged_aqi = self._weighted_average(
            [
                (source.get("air_quality_index"), self._source_weight(source))
                for source in candidates
            ]
        )
        merged_pm25 = self._weighted_average(
            [(source.get("pm25_ug_m3"), self._source_weight(source)) for source in candidates]
        )
        if merged_aqi is None and merged_pm25 is None:
            return None

        best_source = max(candidates, key=self._source_weight)
        sources = [
            str(source.get("air_quality_source"))
            for source in candidates
            if source.get("air_quality_source")
        ]
        overall_confidence = self._weighted_average(
            [
                (source.get("source_confidence"), self._source_weight(source))
                for source in candidates
            ]
        )

        return {
            "air_quality_index": int(round(merged_aqi)) if merged_aqi is not None else None,
            "pm25_ug_m3": merged_pm25,
            "air_quality_category": self._aqi_category(merged_aqi),
            "air_quality_source": " + ".join(sorted(set(sources)))
            if len(sources) > 1
            else sources[0],
            "air_quality_observed_at": best_source.get("air_quality_observed_at"),
            "air_quality_confidence": overall_confidence,
            "air_quality_sources": sources,
        }

    def _fetch_openmeteo(
        self,
        client: httpx.Client,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any] | None:
        response = client.get(
            self.base_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "us_aqi,pm2_5",
                "timezone": "auto",
            },
        )
        response.raise_for_status()

        payload = response.json()
        current = payload.get("current") or {}
        aqi = self._to_float(current.get("us_aqi"))
        pm25 = self._to_float(current.get("pm2_5"))
        if aqi is None and pm25 is None:
            return None

        observed_at = self._parse_observed_at(
            current.get("time"),
            payload.get("utc_offset_seconds"),
        )
        return {
            "air_quality_index": aqi,
            "pm25_ug_m3": pm25,
            "air_quality_source": "Open-Meteo Air Quality",
            "air_quality_observed_at": observed_at,
            "source_confidence": 0.65,
        }

    def _fetch_airnow(
        self,
        client: httpx.Client,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any] | None:
        if not self.airnow_api_key:
            return None

        try:
            response = client.get(
                self.airnow_base_url,
                params={
                    "format": "application/json",
                    "latitude": latitude,
                    "longitude": longitude,
                    "distance": self.airnow_radius_miles,
                    "API_KEY": self.airnow_api_key,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("AirNow fetch failed near (%s, %s): %s", latitude, longitude, exc)
            return None

        rows = response.json()
        if not isinstance(rows, list) or not rows:
            return None

        strongest_row = None
        strongest_aqi = None
        pm25 = None

        for row in rows:
            if not isinstance(row, dict):
                continue
            row_aqi = self._to_float(row.get("AQI"))
            if row_aqi is not None and (strongest_aqi is None or row_aqi > strongest_aqi):
                strongest_aqi = row_aqi
                strongest_row = row

            parameter_name = str(row.get("ParameterName") or "").upper()
            if "PM2.5" in parameter_name or parameter_name == "PM25":
                concentration = self._to_float(row.get("Concentration"))
                if concentration is not None:
                    pm25 = concentration

        if strongest_aqi is None and pm25 is None:
            return None

        observed_at = self._parse_airnow_observed_at(strongest_row or {})
        return {
            "air_quality_index": strongest_aqi,
            "pm25_ug_m3": pm25,
            "air_quality_source": "EPA AirNow",
            "air_quality_observed_at": observed_at,
            "source_confidence": 0.9,
        }

    @staticmethod
    def _weighted_average(weighted_values: list[tuple[object, float]]) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for value, weight in weighted_values:
            numeric = AirQualityService._to_float(value)
            if numeric is None or weight <= 0:
                continue
            numerator += numeric * weight
            denominator += weight

        if denominator <= 0:
            return None
        return numerator / denominator

    @staticmethod
    def _source_weight(source: dict[str, Any]) -> float:
        base = AirQualityService._to_float(source.get("source_confidence")) or 0.5
        observed_at = source.get("air_quality_observed_at")
        if not isinstance(observed_at, datetime):
            return base * 0.8

        now = datetime.now(UTC)
        age_hours = max(0.0, (now - observed_at).total_seconds() / 3600.0)
        if age_hours <= 1.0:
            freshness = 1.0
        elif age_hours <= 6.0:
            freshness = 0.9
        elif age_hours <= 24.0:
            freshness = 0.7
        else:
            freshness = 0.5
        return base * freshness

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _aqi_category(aqi: float | None) -> str | None:
        if aqi is None:
            return None
        if aqi <= 50:
            return "Good"
        if aqi <= 100:
            return "Moderate"
        if aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        if aqi <= 200:
            return "Unhealthy"
        if aqi <= 300:
            return "Very Unhealthy"
        return "Hazardous"

    @staticmethod
    def _parse_observed_at(value: Any, utc_offset_seconds: Any) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            observed = datetime.fromisoformat(value)
        except ValueError:
            return None

        if observed.tzinfo is not None:
            return observed.astimezone(UTC)

        offset_seconds = int(utc_offset_seconds or 0)
        return (observed - timedelta(seconds=offset_seconds)).replace(tzinfo=UTC)

    @staticmethod
    def _parse_airnow_observed_at(row: dict[str, Any]) -> datetime | None:
        date_observed = row.get("DateObserved")
        hour_observed = row.get("HourObserved")
        if not isinstance(date_observed, str):
            return None
        try:
            observed_date = datetime.fromisoformat(date_observed.strip()).date()
        except ValueError:
            return None

        try:
            hour = int(hour_observed or 0)
        except (TypeError, ValueError):
            hour = 0

        return datetime(
            observed_date.year,
            observed_date.month,
            observed_date.day,
            hour,
            0,
            0,
            tzinfo=UTC,
        )
