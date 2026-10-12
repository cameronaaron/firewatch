"""
Satellite and remote sensing data service for fire detection.

Uses authenticated NASA FIRMS active-fire detections together with real
STAC-derived vegetation and thermal products from the Planetary Computer.
"""

import asyncio
import csv
import io
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class FireDetection:
    """Satellite fire detection result."""

    latitude: float
    longitude: float
    confidence: float
    sensor: str
    timestamp: datetime
    brightness_temp: float | None = None
    radiative_power: float | None = None
    cloud_coverage: float | None = None


class SatelliteFireService:
    """Satellite fire detection service backed by FIRMS and STAC raster products."""

    _FIRMS_MIN_DAY_RANGE = 1
    _FIRMS_MAX_DAY_RANGE = 5
    _DEFAULT_FIRMS_SOURCES = (
        "VIIRS_NOAA21_NRT",
        "VIIRS_NOAA20_NRT",
        "VIIRS_SNPP_NRT",
        "MODIS_NRT",
    )
    _SENTINEL_MASK_CLASSES = {0, 1, 3, 6, 8, 9, 10, 11}
    _LANDSAT_QA_CLOUD_BITS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)

    def __init__(self):
        """Initialize satellite service."""
        self.firms_base = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
        self.planetary_computer = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.planetary_computer_sas = "https://planetarycomputer.microsoft.com/api/sas/v1/token"

        self.firms_map_key = settings.FIRMS_MAP_KEY
        self.firms_sources = tuple(settings.FIRMS_SOURCE_IDS or self._DEFAULT_FIRMS_SOURCES)
        self.firms_timeout = settings.FIRMS_API_TIMEOUT
        self.sample_radius_pixels = settings.SATELLITE_SAMPLE_RADIUS_PIXELS

        self._signed_token_cache: dict[str, tuple[str, datetime]] = {}

    async def get_firms_detections(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 50,
        days: int = 5,
    ) -> list[FireDetection]:
        """Get active-fire detections from authenticated NASA FIRMS sources."""
        if not self.firms_map_key:
            logger.info(
                "Skipping FIRMS detections near (%s, %s); FIRMS_MAP_KEY is not configured",
                latitude,
                longitude,
            )
            return []

        bbox = self._bbox_from_radius_km(latitude, longitude, radius_km)
        normalized_days = self._normalize_firms_day_range(days)
        results = await asyncio.gather(
            *[
                self._fetch_firms_area_source(source, bbox, normalized_days)
                for source in self.firms_sources
            ],
            return_exceptions=True,
        )

        detections: list[FireDetection] = []
        for source, result in zip(self.firms_sources, results, strict=False):
            if isinstance(result, BaseException):
                logger.warning("FIRMS source %s failed: %s", source, result)
                continue
            detections.extend(result)

        deduplicated = self._deduplicate_detections(detections)
        deduplicated.sort(
            key=lambda item: (
                item.timestamp,
                item.confidence,
                item.radiative_power or 0.0,
            ),
            reverse=True,
        )
        return deduplicated

    async def get_modis_detections(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 50,
        days: int = 5,
    ) -> list[FireDetection]:
        """Backward-compatible alias for the FIRMS multi-sensor active-fire feed."""
        return await self.get_firms_detections(latitude, longitude, radius_km, days)

    async def get_sentinel2_ndvi(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Get Sentinel-2 vegetation products derived from local raster samples."""
        if start_date is None:
            start_date = datetime.now(UTC) - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now(UTC)

        try:
            item = await self._search_planetary_computer(
                collections=["sentinel-2-l2a"],
                bbox=self._bbox_around_point(latitude, longitude, 0.05),
                start_date=start_date,
                end_date=end_date,
                query={"eo:cloud_cover": {"lte": 30}},
            )
            if not item:
                return None

            properties = item.get("properties", {})
            assets = item.get("assets", {})
            no_data_pct = self._to_float(properties.get("s2:nodata_pixel_percentage"))

            result: dict[str, Any] = {
                "source": "Sentinel-2",
                "scene_id": item.get("id"),
                "collection": item.get("collection"),
                "observation_time": properties.get("datetime"),
                "cloud_coverage": self._to_float(properties.get("eo:cloud_cover")),
                "valid_data_pct": None if no_data_pct is None else max(0.0, 100.0 - no_data_pct),
                "nir_red_assets_available": all(band in assets for band in ("B04", "B08")),
                "asset_count": len(assets),
            }
            result.update(await self._derive_sentinel_vegetation_product(item, latitude, longitude))
            return result
        except Exception as e:
            logger.warning("Sentinel-2 vegetation fetch failed: %s", e)
            return None

    async def get_landsat_thermal(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any] | None:
        """Get Landsat thermal products derived from local raster samples."""
        try:
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=60)
            item = await self._search_planetary_computer(
                collections=["landsat-c2-l2"],
                bbox=self._bbox_around_point(latitude, longitude, 0.08),
                start_date=start_date,
                end_date=end_date,
                query={"eo:cloud_cover": {"lte": 40}},
            )
            if not item:
                return None

            properties = item.get("properties", {})
            assets = item.get("assets", {})
            thermal_assets = [
                key
                for key in assets
                if key.lower() in {"lwir11", "lwir"} or "thermal" in key.lower()
            ]

            result: dict[str, Any] = {
                "source": "Landsat",
                "scene_id": item.get("id"),
                "collection": item.get("collection"),
                "observation_time": properties.get("datetime"),
                "cloud_coverage": self._to_float(properties.get("eo:cloud_cover")),
                "thermal_asset_keys": thermal_assets,
                "thermal_asset_count": len(thermal_assets),
            }
            result.update(await self._derive_landsat_thermal_product(item, latitude, longitude))
            return result
        except Exception as e:
            logger.warning("Landsat thermal fetch failed: %s", e)
            return None

    async def _fetch_firms_area_source(
        self,
        source: str,
        bbox: tuple[float, float, float, float],
        days: int,
    ) -> list[FireDetection]:
        """Fetch one FIRMS source using the authenticated area CSV endpoint."""
        url = self._build_firms_area_url(source, bbox, days)
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=self.firms_timeout)
            response.raise_for_status()
            return self._parse_firms_csv(response.text, source)

    def _build_firms_area_url(
        self,
        source: str,
        bbox: tuple[float, float, float, float],
        days: int,
    ) -> str:
        """Build the FIRMS authenticated area URL for a bounding box."""
        west, south, east, north = bbox
        bbox_text = f"{west:.5f},{south:.5f},{east:.5f},{north:.5f}"
        normalized_days = self._normalize_firms_day_range(days)
        return f"{self.firms_base}/{self.firms_map_key}/{source}/{bbox_text}/{normalized_days}"

    def _parse_firms_csv(self, csv_text: str, source: str) -> list[FireDetection]:
        """Parse FIRMS CSV rows into normalized fire detections."""
        if not csv_text.strip():
            return []

        detections: list[FireDetection] = []
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            latitude = self._to_float(row.get("latitude"))
            longitude = self._to_float(row.get("longitude"))
            if latitude is None or longitude is None:
                continue

            detections.append(
                FireDetection(
                    latitude=latitude,
                    longitude=longitude,
                    confidence=self._parse_firms_confidence(row.get("confidence")),
                    sensor=row.get("instrument") or row.get("satellite") or source,
                    timestamp=self._parse_firms_timestamp(row),
                    brightness_temp=self._to_float(
                        row.get("bright_ti4") or row.get("brightness") or row.get("bright_t31")
                    ),
                    radiative_power=self._to_float(row.get("frp")),
                )
            )

        return detections

    async def _derive_sentinel_vegetation_product(
        self,
        item: dict[str, Any],
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Derive NDVI/EVI summaries from Sentinel-2 raster windows."""
        assets = item.get("assets", {})
        collection_id = item.get("collection") or "sentinel-2-l2a"
        if not all(key in assets for key in ("B04", "B08")):
            return {}

        red_task = self._load_asset_window(
            collection_id,
            assets["B04"],
            latitude,
            longitude,
            reflectance_scale=1 / 10000.0,
        )
        nir_task = self._load_asset_window(
            collection_id,
            assets["B08"],
            latitude,
            longitude,
            reflectance_scale=1 / 10000.0,
        )
        blue_task = self._load_asset_window(
            collection_id,
            assets.get("B02"),
            latitude,
            longitude,
            reflectance_scale=1 / 10000.0,
        )
        scl_task = self._load_asset_window(collection_id, assets.get("SCL"), latitude, longitude)

        red, nir, blue, scl = await asyncio.gather(
            red_task,
            nir_task,
            blue_task,
            scl_task,
        )
        if red is None or nir is None:
            return {}

        red = self._normalize_reflectance(red)
        nir = self._normalize_reflectance(nir)
        mask = np.ma.getmaskarray(red) | np.ma.getmaskarray(nir)
        if scl is not None:
            mask |= np.isin(
                np.asarray(np.ma.filled(scl, fill_value=-1), dtype=int),
                tuple(self._SENTINEL_MASK_CLASSES),
            )

        red_masked = np.ma.array(red, mask=mask)
        nir_masked = np.ma.array(nir, mask=mask)
        ndvi = self._normalized_difference(nir_masked, red_masked)
        ndvi_stats = self._summarize_masked_array(ndvi)
        if not ndvi_stats:
            return {}

        result: dict[str, Any] = {
            "ndvi_mean": ndvi_stats["mean"],
            "ndvi_std": ndvi_stats["std"],
            "ndvi_p90": ndvi_stats["p90"],
            "valid_pixel_count": ndvi_stats["count"],
            "sample_valid_pct": ndvi_stats["valid_pct"],
        }

        if blue is not None:
            blue_masked = np.ma.array(self._normalize_reflectance(blue), mask=mask)
            denominator = nir_masked + (6 * red_masked) - (7.5 * blue_masked) + 1.0
            evi = np.ma.array(
                np.divide(
                    2.5 * (nir_masked - red_masked),
                    denominator,
                    out=np.zeros(denominator.shape, dtype=float),
                    where=np.abs(denominator) > 1e-6,
                ),
                mask=mask | (np.abs(denominator) <= 1e-6),
            )
            evi_stats = self._summarize_masked_array(np.ma.clip(evi, -1.0, 1.5))
            if evi_stats:
                result["evi_mean"] = evi_stats["mean"]

        cloud_coverage = self._to_float(item.get("properties", {}).get("eo:cloud_cover")) or 0.0
        result["confidence"] = max(
            0.0,
            min(1.0, (result["sample_valid_pct"] / 100.0) * (1.0 - cloud_coverage / 100.0)),
        )
        return result

    async def _derive_landsat_thermal_product(
        self,
        item: dict[str, Any],
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Derive land-surface temperature summaries from Landsat raster windows."""
        assets = item.get("assets", {})
        collection_id = item.get("collection") or "landsat-c2-l2"
        thermal_asset = None
        thermal_asset_key = None
        for key in ("lwir11", "lwir"):
            if key in assets:
                thermal_asset = assets[key]
                thermal_asset_key = key
                break

        if thermal_asset is None:
            return {}

        thermal_task = self._load_asset_window(
            collection_id,
            thermal_asset,
            latitude,
            longitude,
            scale_hint=0.00341802,
            offset_hint=149.0,
        )
        qa_task = self._load_asset_window(
            collection_id, assets.get("qa_pixel"), latitude, longitude
        )
        thermal_data, qa_pixel = await asyncio.gather(thermal_task, qa_task)
        if thermal_data is None:
            return {}

        thermal_mask = np.ma.getmaskarray(thermal_data)
        if qa_pixel is not None:
            qa_values = np.asarray(np.ma.filled(qa_pixel, fill_value=0), dtype=np.uint16)
            thermal_mask |= (qa_values & self._LANDSAT_QA_CLOUD_BITS) != 0

        thermal = np.ma.array(thermal_data, mask=thermal_mask)
        thermal_values = np.asarray(thermal.compressed(), dtype=float)
        if thermal_values.size and np.nanmedian(thermal_values) > 500:
            thermal = np.ma.array((thermal * 0.00341802) + 149.0, mask=thermal_mask)

        thermal_stats = self._summarize_masked_array(thermal)
        if not thermal_stats:
            return {}

        thermal_values = np.asarray(thermal.compressed(), dtype=float)
        hotspot_threshold = max(330.0, float(np.percentile(thermal_values, 95)))
        hotspot_fraction = (
            float(np.mean(thermal_values >= hotspot_threshold)) if thermal_values.size else 0.0
        )

        return {
            "land_surface_temp_k": thermal_stats["mean"],
            "land_surface_temp_c": thermal_stats["mean"] - 273.15,
            "thermal_p90_k": thermal_stats["p90"],
            "thermal_hotspot_fraction": hotspot_fraction,
            "valid_pixel_count": thermal_stats["count"],
            "sample_valid_pct": thermal_stats["valid_pct"],
            "thermal_asset_key": thermal_asset_key,
        }

    async def _load_asset_window(
        self,
        collection_id: str,
        asset: dict[str, Any] | None,
        latitude: float,
        longitude: float,
        reflectance_scale: float | None = None,
        scale_hint: float | None = None,
        offset_hint: float | None = None,
    ) -> np.ma.MaskedArray | None:
        """Load a small signed raster window around a point."""
        if not asset or not asset.get("href"):
            return None

        signed_href = await self._sign_planetary_computer_asset(collection_id, asset["href"])
        return await asyncio.to_thread(
            self._read_raster_window,
            signed_href,
            latitude,
            longitude,
            self.sample_radius_pixels,
            reflectance_scale,
            scale_hint,
            offset_hint,
        )

    async def _sign_planetary_computer_asset(self, collection_id: str, asset_href: str) -> str:
        """Attach a SAS token to Planetary Computer asset URLs when needed."""
        if "?" in asset_href:
            return asset_href

        token = await self._get_planetary_computer_token(collection_id)
        return f"{asset_href}?{token}"

    async def _get_planetary_computer_token(self, collection_id: str) -> str:
        """Fetch and cache a Planetary Computer SAS token for a collection."""
        cached = self._signed_token_cache.get(collection_id)
        now = datetime.now(UTC)
        if cached and cached[1] > now:
            return cached[0]

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.planetary_computer_sas}/{collection_id}",
                timeout=10.0,
            )
            response.raise_for_status()
            token = response.json().get("token")

        if not token:
            raise ValueError(f"Planetary Computer SAS token missing for collection {collection_id}")

        self._signed_token_cache[collection_id] = (token, now + timedelta(minutes=45))
        return str(token)

    def _read_raster_window(
        self,
        href: str,
        latitude: float,
        longitude: float,
        sample_radius_pixels: int,
        reflectance_scale: float | None,
        scale_hint: float | None,
        offset_hint: float | None,
    ) -> np.ma.MaskedArray | None:
        """Read a small raster window centered on the input point."""
        try:
            import rasterio
            from rasterio.warp import transform
            from rasterio.windows import Window
        except ImportError:
            logger.warning("rasterio is unavailable; satellite raster products cannot be derived")
            return None

        env_kwargs = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".TIF,.tif,.TIFF,.tiff",
        }

        try:
            with rasterio.Env(**env_kwargs):
                with rasterio.open(href) as dataset:
                    if dataset.crs is None:
                        return None

                    xs, ys = transform("EPSG:4326", dataset.crs, [longitude], [latitude])
                    row, col = dataset.index(xs[0], ys[0])
                    if row < 0 or col < 0 or row >= dataset.height or col >= dataset.width:
                        return None

                    row_off = max(0, row - sample_radius_pixels)
                    col_off = max(0, col - sample_radius_pixels)
                    row_end = min(dataset.height, row + sample_radius_pixels + 1)
                    col_end = min(dataset.width, col + sample_radius_pixels + 1)
                    window = Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=col_end - col_off,
                        height=row_end - row_off,
                    )

                    data = dataset.read(1, window=window, masked=True).astype("float32")
                    if data.size == 0:
                        return None

                    scale = dataset.scales[0] if getattr(dataset, "scales", None) else 1.0
                    offset = dataset.offsets[0] if getattr(dataset, "offsets", None) else 0.0

                    if reflectance_scale is not None and scale == 1.0 and offset == 0.0:
                        scale = reflectance_scale
                    elif scale_hint is not None and scale == 1.0 and offset == 0.0:
                        scale = scale_hint
                        offset = offset_hint or 0.0

                    if scale != 1.0 or offset != 0.0:
                        data = (data * scale) + offset

                    return data
        except Exception as exc:
            logger.warning("Raster read failed for %s: %s", href, exc)
            return None

    async def _search_planetary_computer(
        self,
        collections: list[str],
        bbox: list[float],
        start_date: datetime,
        end_date: datetime,
        query: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Search Planetary Computer STAC and return the most recent item."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.planetary_computer}/search",
                json={
                    "collections": collections,
                    "bbox": bbox,
                    "datetime": f"{start_date.isoformat()}/{end_date.isoformat()}",
                    "query": query or {},
                    "sortby": [{"field": "properties.datetime", "direction": "desc"}],
                    "limit": 1,
                },
                timeout=15.0,
            )
            if response.status_code != 200:
                return None

            data = response.json()
            features = data.get("features", [])
            if not isinstance(features, list) or not features:
                return None

            first = features[0]
            return first if isinstance(first, dict) else None

    @classmethod
    def _bbox_from_radius_km(
        cls,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> tuple[float, float, float, float]:
        """Build a geodesic-ish bounding box around a point in kilometers."""
        lat_delta = radius_km / 111.32
        lon_delta = radius_km / max(111.32 * math.cos(math.radians(latitude)), 0.1)
        return (
            longitude - lon_delta,
            latitude - lat_delta,
            longitude + lon_delta,
            latitude + lat_delta,
        )

    @staticmethod
    def _bbox_around_point(latitude: float, longitude: float, radius_deg: float) -> list[float]:
        """Build a small bbox around a point for STAC search."""
        return [
            longitude - radius_deg,
            latitude - radius_deg,
            longitude + radius_deg,
            latitude + radius_deg,
        ]

    @staticmethod
    def _normalized_difference(a: np.ma.MaskedArray, b: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """Compute a normalized-difference index from two bands."""
        denominator = a + b
        return np.ma.array(
            np.divide(
                a - b,
                denominator,
                out=np.zeros(denominator.shape, dtype=float),
                where=np.abs(denominator) > 1e-6,
            ),
            mask=np.ma.getmaskarray(a) | np.ma.getmaskarray(b) | (np.abs(denominator) <= 1e-6),
        )

    @staticmethod
    def _normalize_reflectance(data: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """Convert scaled reflectance arrays into unitless reflectance when needed."""
        values = np.asarray(data.compressed(), dtype=float)
        if values.size and np.nanpercentile(values, 95) > 2.0:
            return np.ma.array(data / 10000.0, mask=np.ma.getmaskarray(data))
        return data

    @staticmethod
    def _summarize_masked_array(data: np.ma.MaskedArray) -> dict[str, Any]:
        """Summarize a masked raster window with robust numeric stats."""
        values = np.asarray(data.compressed(), dtype=float)
        total_pixels = int(data.size)
        if values.size == 0 or total_pixels == 0:
            return {}

        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p90": float(np.percentile(values, 90)),
            "count": int(values.size),
            "valid_pct": float((values.size / total_pixels) * 100.0),
        }

    @classmethod
    def _deduplicate_detections(cls, detections: list[FireDetection]) -> list[FireDetection]:
        """Remove duplicate FIRMS rows while keeping the highest-confidence record."""
        deduplicated: dict[tuple[float, float, str, str], FireDetection] = {}
        for detection in detections:
            key = (
                round(detection.latitude, 3),
                round(detection.longitude, 3),
                detection.timestamp.isoformat(),
                detection.sensor,
            )
            current = deduplicated.get(key)
            if current is None or detection.confidence >= current.confidence:
                deduplicated[key] = detection
        return list(deduplicated.values())

    @classmethod
    def _normalize_firms_day_range(cls, days: int) -> int:
        """Clamp FIRMS day ranges to NASA's supported 1..5 window for the area API."""
        return max(cls._FIRMS_MIN_DAY_RANGE, min(cls._FIRMS_MAX_DAY_RANGE, int(days)))

    @staticmethod
    def _parse_firms_confidence(value: Any) -> float:
        """Normalize FIRMS confidence values to a 0-100 scale."""
        if value is None:
            return 0.0

        text = str(value).strip().lower()
        if text in {"l", "low"}:
            return 35.0
        if text in {"n", "nominal", "medium"}:
            return 60.0
        if text in {"h", "high"}:
            return 85.0

        numeric = SatelliteFireService._to_float(value)
        return numeric if numeric is not None else 0.0

    @staticmethod
    def _parse_firms_timestamp(row: dict[str, Any]) -> datetime:
        """Parse FIRMS acquisition timestamps into UTC datetimes."""
        date_text = str(row.get("acq_date") or row.get("date") or "").strip()
        time_text = str(row.get("acq_time") or row.get("time") or "").strip()
        if date_text:
            try:
                date_value = datetime.fromisoformat(date_text)
                if time_text.isdigit():
                    time_text = time_text.zfill(4)
                    hour = min(23, int(time_text[:2]))
                    minute = min(59, int(time_text[2:4]))
                    return datetime(
                        date_value.year,
                        date_value.month,
                        date_value.day,
                        hour,
                        minute,
                        tzinfo=UTC,
                    )
                return datetime(
                    date_value.year,
                    date_value.month,
                    date_value.day,
                    tzinfo=UTC,
                )
            except ValueError:
                pass

        return datetime.now(UTC)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Convert a value to float when possible."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class TopographicService:
    """Topographic analysis using source-backed elevation metadata."""

    def __init__(self):
        """Initialize topographic service."""
        self.usgs_dem_base = (
            "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
        )

    async def get_elevation_and_slope(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Get elevation metadata from USGS 3DEP."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.usgs_dem_base}/identify",
                    params={
                        "geometry": f'{{"x":{longitude},"y":{latitude}}}',
                        "geometryType": "esriGeometryPoint",
                        "f": "json",
                    },
                    timeout=10.0,
                )
                if response.status_code != 200:
                    return {}

                data = response.json()
                elevation = SatelliteFireService._to_float(data.get("value"))
                if elevation is None:
                    return {}

                return {
                    "elevation_m": elevation,
                    "slope_deg": None,
                    "aspect_deg": None,
                    "source": "USGS_3DEP",
                }
        except Exception as e:
            logger.warning("Elevation fetch failed: %s", e)

        return {}

    async def get_fuel_type(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Return no fabricated fuel classification when no raster fuel map is configured."""
        logger.info(
            "Skipping fuel classification near (%s, %s); raster land-cover sampling is not "
            "configured",
            latitude,
            longitude,
        )
        return {}


class VegetationService:
    """Vegetation analysis backed by real Sentinel-2 and Landsat products."""

    def __init__(self):
        """Initialize vegetation service."""
        self.satellite_service = SatelliteFireService()

    async def get_ndvi(
        self,
        latitude: float,
        longitude: float,
        days_back: int = 7,
    ) -> dict[str, Any]:
        """Get real NDVI and EVI summaries when Sentinel-2 imagery is available."""
        try:
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=days_back)
            scene = await self.satellite_service.get_sentinel2_ndvi(
                latitude,
                longitude,
                start_date=start_date,
                end_date=end_date,
            )
            if not scene:
                return {}

            return {
                "source": scene.get("source"),
                "scene_id": scene.get("scene_id"),
                "ndvi": scene.get("ndvi_mean"),
                "ndvi_p90": scene.get("ndvi_p90"),
                "evi": scene.get("evi_mean"),
                "cloud_coverage_pct": scene.get("cloud_coverage"),
                "observation_time": scene.get("observation_time"),
            }
        except Exception as e:
            logger.warning("NDVI fetch failed: %s", e)
            return {}

    async def get_moisture_stress(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Estimate vegetation moisture stress from real NDVI and LST products."""
        try:
            sentinel_task = self.satellite_service.get_sentinel2_ndvi(latitude, longitude)
            thermal_task = self.satellite_service.get_landsat_thermal(latitude, longitude)
            sentinel_scene, thermal_scene = await asyncio.gather(sentinel_task, thermal_task)

            if not sentinel_scene and not thermal_scene:
                return {}

            ndvi = SatelliteFireService._to_float(
                None if not sentinel_scene else sentinel_scene.get("ndvi_mean")
            )
            land_surface_temp_k = SatelliteFireService._to_float(
                None if not thermal_scene else thermal_scene.get("land_surface_temp_k")
            )
            moisture_stress = self._calculate_moisture_stress_proxy(ndvi, land_surface_temp_k)

            result = {
                "source": "+".join(
                    source
                    for source in (
                        None if not sentinel_scene else sentinel_scene.get("source"),
                        None if not thermal_scene else thermal_scene.get("source"),
                    )
                    if source
                )
                or None,
                "ndvi": ndvi,
                "land_surface_temp_k": land_surface_temp_k,
                "land_surface_temp_c": None
                if land_surface_temp_k is None
                else land_surface_temp_k - 273.15,
                "thermal_hotspot_fraction": None
                if not thermal_scene
                else thermal_scene.get("thermal_hotspot_fraction"),
                "moisture_stress_index": moisture_stress,
                "observation_time": (
                    None if not thermal_scene else thermal_scene.get("observation_time")
                )
                or (None if not sentinel_scene else sentinel_scene.get("observation_time")),
            }
            return {key: value for key, value in result.items() if value is not None}
        except Exception as e:
            logger.warning("Moisture stress calculation failed: %s", e)
            return {}

    @staticmethod
    def _calculate_moisture_stress_proxy(
        ndvi: float | None,
        land_surface_temp_k: float | None,
    ) -> float | None:
        """Compute a bounded moisture-stress proxy from real vegetation and thermal inputs."""
        if ndvi is None or land_surface_temp_k is None:
            return None

        land_surface_temp_c = land_surface_temp_k - 273.15
        vegetation_stress = np.clip((0.75 - ndvi) / 0.75, 0.0, 1.0)
        thermal_stress = np.clip((land_surface_temp_c - 20.0) / 25.0, 0.0, 1.0)
        return float(np.round((0.6 * vegetation_stress) + (0.4 * thermal_stress), 4))
