"""Current weather and forecasts merged from NOAA/NWS, Open-Meteo and Meteostat."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from app.ml.fire_weather import danger_level, fosberg_index

logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    """Comprehensive weather data structure."""

    timestamp: datetime
    temperature_f: float | None
    humidity_pct: float | None
    wind_speed_mph: float | None
    wind_direction_deg: int | None
    precipitation_in: float | None
    pressure_mb: float | None
    dewpoint_f: float | None
    visibility_miles: float | None
    cloud_coverage_pct: int | None
    solar_radiation_w_m2: float | None = None
    source: str = "merged"  # Data source identifier


class AdvancedWeatherService:
    """Fetches each source independently and blends what they report."""

    def __init__(self):
        self.noaa_base = "https://api.weather.gov"
        self.openmeteo_base = "https://api.open-meteo.com/v1"
        self.meteostat_base = "https://api.meteostat.net"

    async def get_noaa_weather(self, latitude: float, longitude: float) -> dict[str, Any] | None:
        """
        Get weather from NOAA/NWS (National Weather Service).

        Most authoritative source for US locations.
        Includes detailed forecasts and hazardous weather info.
        """
        try:
            async with httpx.AsyncClient() as client:
                # First, get grid point
                grid_resp = await client.get(
                    f"{self.noaa_base}/points/{latitude:.4f},{longitude:.4f}", timeout=10.0
                )
                if grid_resp.status_code != 200:
                    return None

                grid_data = grid_resp.json()
                forecast_url = grid_data.get("properties", {}).get("forecast")

                if not forecast_url:
                    return None

                # Get detailed forecast
                forecast_resp = await client.get(forecast_url, timeout=10.0)
                if forecast_resp.status_code != 200:
                    return None

                forecast = forecast_resp.json()
                current = forecast.get("properties", {}).get("periods", [{}])[0]

                return {
                    "source": "NOAA",
                    "temperature_f": current.get("temperature"),
                    "wind_speed_mph": int(current.get("windSpeed", "0 mph").split()[0]),
                    "wind_direction": current.get("windDirection", "N"),
                    "wind_direction_deg": self._direction_to_degrees(
                        current.get("windDirection", "N")
                    ),
                    "short_forecast": current.get("shortForecast"),
                    "detailed_forecast": current.get("detailedForecast"),
                    "hazards": current.get("hazards", []),
                }
        except Exception as e:
            logger.warning(f"NOAA fetch failed: {e}")
            return None

    async def get_openmeteo_weather(
        self, latitude: float, longitude: float
    ) -> dict[str, Any] | None:
        """Get current conditions from Open-Meteo for dense numeric coverage."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.openmeteo_base}/forecast",
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "current": (
                            "temperature_2m,relative_humidity_2m,precipitation,"
                            "wind_speed_10m,wind_direction_10m,surface_pressure,"
                            "dew_point_2m,visibility,cloud_cover"
                        ),
                        "temperature_unit": "fahrenheit",
                        "wind_speed_unit": "mph",
                        "precipitation_unit": "inch",
                        "timezone": "auto",
                    },
                    timeout=10.0,
                )
                if resp.status_code != 200:
                    return None

                data = resp.json()
                current = data.get("current", {})
                return {
                    "source": "Open-Meteo",
                    "temperature_f": current.get("temperature_2m"),
                    "humidity_pct": current.get("relative_humidity_2m"),
                    "wind_speed_mph": current.get("wind_speed_10m"),
                    "wind_direction_deg": current.get("wind_direction_10m"),
                    "precipitation_in": current.get("precipitation"),
                    "pressure_mb": current.get("surface_pressure"),
                    "dewpoint_f": current.get("dew_point_2m"),
                    "visibility_miles": (
                        current.get("visibility") / 1609.34
                        if current.get("visibility") is not None
                        else None
                    ),
                    "cloud_coverage_pct": current.get("cloud_cover"),
                    "timestamp": current.get("time"),
                }
        except Exception as e:
            logger.warning(f"Open-Meteo fetch failed: {e}")
            return None

    async def get_meteostat_weather(
        self, latitude: float, longitude: float
    ) -> dict[str, Any] | None:
        """
        Get historical weather from Meteostat.

        Provides high-quality historical data for analysis.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.meteostat_base}/point/latest",
                    params={
                        "lat": latitude,
                        "lon": longitude,
                    },
                    timeout=10.0,
                )
                if resp.status_code != 200:
                    return None

                data = resp.json()
                current = data.get("data", {})

                return {
                    "source": "Meteostat",
                    "temperature_f": current.get("temp") * 9 / 5 + 32
                    if current.get("temp")
                    else None,
                    "humidity_pct": current.get("rhum"),
                    "wind_speed_mph": current.get("wspd") * 0.621371
                    if current.get("wspd")
                    else None,
                    "wind_direction_deg": current.get("wdir"),
                    "precipitation_in": current.get("prcp") / 25.4 if current.get("prcp") else None,
                    "pressure_mb": current.get("pres"),
                    "dewpoint_f": current.get("dwpt") * 9 / 5 + 32 if current.get("dwpt") else None,
                    "visibility_miles": current.get("vvis") / 1609.34
                    if current.get("vvis")
                    else None,
                }
        except Exception as e:
            logger.warning(f"Meteostat fetch failed: {e}")
            return None

    async def merge_weather_sources(self, latitude: float, longitude: float) -> WeatherData | None:
        """
        Merge data from multiple sources using intelligent weighting.

        Prioritizes accuracy and freshness:
        1. NOAA (best for US)
        2. Meteostat (historical accuracy)
        3. Open-Meteo (fallback)
        """
        tasks = [
            self.get_noaa_weather(latitude, longitude),
            self.get_openmeteo_weather(latitude, longitude),
            self.get_meteostat_weather(latitude, longitude),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        noaa_data = results[0] if isinstance(results[0], dict) else None
        openmeteo_data = results[1] if isinstance(results[1], dict) else None
        meteo_data = results[2] if isinstance(results[2], dict) else None

        available_sources = [source for source in (noaa_data, openmeteo_data, meteo_data) if source]

        if available_sources:
            source_weights = {
                "NOAA": 0.45,
                "Open-Meteo": 0.40,
                "Meteostat": 0.15,
            }

            return WeatherData(
                timestamp=datetime.now(UTC),
                temperature_f=self._blend_numeric(
                    available_sources, "temperature_f", source_weights
                ),
                humidity_pct=self._blend_numeric(available_sources, "humidity_pct", source_weights),
                wind_speed_mph=self._blend_numeric(
                    available_sources, "wind_speed_mph", source_weights
                ),
                wind_direction_deg=self._preferred_direction(available_sources),
                precipitation_in=self._blend_numeric(
                    available_sources, "precipitation_in", source_weights
                ),
                pressure_mb=self._blend_numeric(available_sources, "pressure_mb", source_weights),
                dewpoint_f=self._blend_numeric(available_sources, "dewpoint_f", source_weights),
                visibility_miles=self._blend_numeric(
                    available_sources, "visibility_miles", source_weights
                ),
                cloud_coverage_pct=self._optional_int(
                    self._blend_numeric(available_sources, "cloud_coverage_pct", source_weights)
                ),
                source="+".join(
                    source["source"] for source in available_sources if source.get("source")
                ),
            )

        return None

    async def get_current_weather(self, latitude: float, longitude: float) -> dict[str, Any] | None:
        """Return merged current conditions using the app's standard weather schema."""
        weather = await self.merge_weather_sources(latitude, longitude)
        if not weather:
            return None

        return {
            "timestamp": weather.timestamp.isoformat(),
            "temperature_f": weather.temperature_f,
            "humidity_pct": weather.humidity_pct,
            "wind_speed_mph": weather.wind_speed_mph,
            "wind_direction_deg": weather.wind_direction_deg,
            "precipitation_in": weather.precipitation_in,
            "pressure_mb": weather.pressure_mb,
            "dewpoint_f": weather.dewpoint_f,
            "visibility_miles": weather.visibility_miles,
            "cloud_coverage_pct": weather.cloud_coverage_pct,
            "source": weather.source,
        }

    async def get_fire_weather_indices(
        self, latitude: float, longitude: float
    ) -> dict[str, Any] | None:
        """Return merged fire weather indices using the app's standard schema."""
        weather = await self.merge_weather_sources(latitude, longitude)
        if not weather:
            return None

        if (
            weather.temperature_f is None
            or weather.humidity_pct is None
            or weather.wind_speed_mph is None
        ):
            return None

        indices = self.calculate_fire_weather_index_v2(weather)
        return {
            "fwi_index": float(indices.get("fwi", 0.0)),
            "danger_level": indices.get("danger_level"),
            "raw_weather": {
                "temperature_f": weather.temperature_f,
                "humidity_pct": weather.humidity_pct,
                "wind_speed_mph": weather.wind_speed_mph,
                "wind_direction_deg": weather.wind_direction_deg,
                "precipitation_in": weather.precipitation_in,
                "source": weather.source,
            },
        }

    async def get_forecast(
        self, latitude: float, longitude: float, days: int = 7
    ) -> dict[str, Any] | None:
        """Return an Open-Meteo-backed forecast using the existing application schema."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.openmeteo_base}/forecast",
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "daily": (
                            "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                            "wind_speed_10m_max,relative_humidity_2m_mean"
                        ),
                        "temperature_unit": "fahrenheit",
                        "wind_speed_unit": "mph",
                        "precipitation_unit": "inch",
                        "timezone": "auto",
                        "forecast_days": min(days, 16),
                    },
                    timeout=10.0,
                )
                if response.status_code != 200:
                    return None

                data = response.json()
                daily = data.get("daily", {})
                forecast_days = []

                for i in range(len(daily.get("time", []))):
                    forecast_days.append(
                        {
                            "date": daily["time"][i],
                            "temp_max_f": daily["temperature_2m_max"][i],
                            "temp_min_f": daily["temperature_2m_min"][i],
                            "precipitation_in": daily["precipitation_sum"][i],
                            "wind_speed_mph": daily["wind_speed_10m_max"][i],
                            "humidity_pct": daily["relative_humidity_2m_mean"][i],
                        }
                    )

                return {
                    "location": {
                        "latitude": latitude,
                        "longitude": longitude,
                    },
                    "forecast": forecast_days,
                }
        except Exception as e:
            logger.warning(f"Forecast fetch failed: {e}")
            return None

    def calculate_fire_weather_index_v2(self, weather: WeatherData) -> dict[str, Any]:
        """Fosberg Fire Weather Index and its danger band, or Nones without inputs."""
        if (
            weather.temperature_f is None
            or weather.humidity_pct is None
            or weather.wind_speed_mph is None
        ):
            return {"fwi": None, "danger_level": None}

        index = fosberg_index(weather.temperature_f, weather.humidity_pct, weather.wind_speed_mph)
        return {"fwi": index, "danger_level": danger_level(index)}

    @staticmethod
    def calculate_fire_spread_multiplier(
        temperature: float,
        humidity: float,
        wind_speed: float,
    ) -> float:
        """Calculate a bounded weather multiplier for spread-rate estimates."""
        if temperature > 85:
            temp_factor = 1 + ((temperature - 85) / 100)
        else:
            temp_factor = 1 - ((85 - temperature) / 200)

        if humidity < 40:
            humidity_factor = 1 + ((40 - humidity) / 100)
        else:
            humidity_factor = 1 - ((humidity - 40) / 200)

        wind_factor = 1 + (wind_speed / 50)
        multiplier = temp_factor * humidity_factor * wind_factor
        return max(0.1, min(10.0, multiplier))

    @staticmethod
    def _blend_numeric(
        sources: list[dict[str, Any]],
        field: str,
        weights: dict[str, float],
    ) -> float | None:
        """Blend numeric fields from multiple weather sources using source weights."""
        weighted_values = []
        for source in sources:
            value = source.get(field)
            if value is None:
                continue

            source_name = str(source.get("source") or "")
            weighted_values.append((float(value), weights.get(source_name, 0.1)))

        if not weighted_values:
            return None

        total_weight = sum(weight for _, weight in weighted_values)
        if total_weight <= 0:
            return None

        return sum(value * weight for value, weight in weighted_values) / total_weight

    @classmethod
    def _preferred_direction(cls, sources: list[dict[str, Any]]) -> int | None:
        """Prefer authoritative wind direction, falling back across merged sources."""
        for source in sources:
            if source.get("wind_direction_deg") is not None:
                return int(source["wind_direction_deg"])
            if source.get("wind_direction"):
                return cls._direction_to_degrees(str(source["wind_direction"]))
        return None

    @staticmethod
    def _optional_int(value: float | None) -> int | None:
        """Convert an optional float to an integer when present."""
        if value is None:
            return None
        return int(round(value))

    @staticmethod
    def _direction_to_degrees(direction: str) -> int:
        """Convert cardinal direction to degrees."""
        directions = {
            "N": 0,
            "NNE": 22.5,
            "NE": 45,
            "ENE": 67.5,
            "E": 90,
            "ESE": 112.5,
            "SE": 135,
            "SSE": 157.5,
            "S": 180,
            "SSW": 202.5,
            "SW": 225,
            "WSW": 247.5,
            "W": 270,
            "WNW": 292.5,
            "NW": 315,
            "NNW": 337.5,
        }
        return int(directions.get(direction, 0))
