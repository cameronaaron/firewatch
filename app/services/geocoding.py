"""Geocoding service for location coordinates."""

import asyncio
import logging
import re
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import httpx
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import GeocodeError
from app.models.database import Location

logger = logging.getLogger(__name__)
settings = get_settings()


class GeocodingService:
    """Service for geocoding addresses to coordinates."""

    _ZIP_CODE_PATTERN = re.compile(r"^\d{5}(?:-\d{4})?$")
    _US_STREET_NUMBER_PATTERN = re.compile(r"\d")

    _geolocator: Nominatim | None = None

    @classmethod
    def _get_geolocator(cls) -> Nominatim:
        """Get or create geolocator instance."""
        if cls._geolocator is None:
            cls._geolocator = Nominatim(
                user_agent=settings.GEOCODER_USER_AGENT, timeout=settings.GEOCODER_TIMEOUT
            )
        return cls._geolocator

    @staticmethod
    def _address_fingerprint(address: str) -> str:
        """Return a stable, non-reversible identifier for an address string."""
        normalized = address.strip().lower().encode("utf-8")
        return sha256(normalized).hexdigest()[:12]

    @staticmethod
    def _normalize_address_query(address: str) -> str:
        """Collapse noisy whitespace so free-form address entry is more forgiving."""
        normalized = re.sub(r"\s*,\s*", ", ", address.strip())
        return re.sub(r"\s+", " ", normalized)

    @classmethod
    def _build_query_candidates(
        cls,
        address: str,
    ) -> list[tuple[str | dict[str, str], dict[str, Any], str]]:
        """Generate increasingly specific queries for free-form addresses and ZIP codes."""
        normalized = cls._normalize_address_query(address)
        candidates: list[tuple[str | dict[str, str], dict[str, Any], str]] = [
            (normalized, {"exactly_one": True, "addressdetails": True}, "freeform"),
        ]

        if cls._ZIP_CODE_PATTERN.fullmatch(normalized):
            zip_code = normalized[:5]
            candidates.append(
                (
                    {"postalcode": zip_code, "country": "United States"},
                    {"exactly_one": True, "country_codes": "us", "addressdetails": True},
                    "us-zipcode-structured",
                )
            )
            candidates.append(
                (
                    f"{zip_code}, USA",
                    {"exactly_one": True, "country_codes": "us", "addressdetails": True},
                    "us-zipcode-freeform",
                )
            )
            return candidates

        lowered = normalized.lower()
        if not lowered.endswith((" usa", " united states", ", usa", ", united states")):
            candidates.append(
                (
                    f"{normalized}, USA",
                    {"exactly_one": True, "country_codes": "us", "addressdetails": True},
                    "us-freeform-fallback",
                )
            )

        return candidates

    @staticmethod
    async def _perform_lookup(
        geolocator: Nominatim,
        query: str | dict[str, str],
        options: dict[str, Any],
    ) -> Any:
        """Run blocking geopy lookups off the event loop."""
        try:
            return await asyncio.to_thread(geolocator.geocode, query, **options)
        except TypeError:
            return await asyncio.to_thread(geolocator.geocode, query)

    @classmethod
    def _should_try_census(cls, address: str) -> bool:
        """Use the Census geocoder for US-style addresses after open geocoders fail."""
        normalized = cls._normalize_address_query(address)
        return bool(
            cls._ZIP_CODE_PATTERN.fullmatch(normalized)
            or cls._US_STREET_NUMBER_PATTERN.search(normalized)
        )

    @classmethod
    async def _lookup_with_census(cls, address: str) -> tuple[float, float] | None:
        """Fallback to the US Census geocoder for authoritative US address matches."""
        if not cls._should_try_census(address):
            return None

        async with httpx.AsyncClient(timeout=settings.GEOCODER_TIMEOUT) as client:
            response = await client.get(
                settings.CENSUS_GEOCODER_URL,
                params={
                    "address": cls._normalize_address_query(address),
                    "benchmark": settings.CENSUS_GEOCODER_BENCHMARK,
                    "format": "json",
                },
            )
            response.raise_for_status()

        matches = (response.json().get("result") or {}).get("addressMatches") or []
        if not matches:
            return None

        coordinates = matches[0].get("coordinates") or {}
        latitude = coordinates.get("y")
        longitude = coordinates.get("x")
        if latitude is None or longitude is None:
            return None
        return float(latitude), float(longitude)

    @classmethod
    async def geocode_location(cls, address: str) -> tuple[float, float]:
        """
        Geocode an address to latitude and longitude.

        Returns:
            Tuple of (latitude, longitude)

        Raises:
            GeocodeError: If geocoding fails
        """
        fingerprint = cls._address_fingerprint(address)
        try:
            geolocator = cls._get_geolocator()
            for query, options, strategy in cls._build_query_candidates(address):
                location = await cls._perform_lookup(geolocator, query, options)
                if not location:
                    continue

                logger.info(
                    "Geocoded address fingerprint=%s via %s to (%.4f, %.4f)",
                    fingerprint,
                    strategy,
                    location.latitude,
                    location.longitude,
                )
                return location.latitude, location.longitude

            census_match = await cls._lookup_with_census(address)
            if census_match is not None:
                logger.info(
                    "Geocoded address fingerprint=%s via census-fallback to (%.4f, %.4f)",
                    fingerprint,
                    census_match[0],
                    census_match[1],
                )
                return census_match

            raise GeocodeError(address)

        except GeocoderTimedOut:
            logger.error("Geocoding timed out for address fingerprint=%s", fingerprint)
            raise GeocodeError(f"Geocoding timed out for: {address}") from None
        except GeocoderUnavailable:
            logger.error("Geocoder unavailable for address fingerprint=%s", fingerprint)
            raise GeocodeError(f"Geocoder unavailable for: {address}") from None
        except httpx.HTTPError as e:
            logger.error(
                "Census geocoder failed for address fingerprint=%s: %s",
                fingerprint,
                e,
            )
            raise GeocodeError(address) from e
        except GeocodeError:
            raise
        except Exception as e:
            logger.error(
                "Unexpected error geocoding address fingerprint=%s: %s",
                fingerprint,
                e,
            )
            raise GeocodeError(address) from e

    @classmethod
    async def geocode_locations(cls, db: Session) -> int:
        """Geocode all active locations that haven't been geocoded yet."""
        locations = (
            db.query(Location)
            .filter(Location.is_active, Location.latitude.is_(None) | Location.longitude.is_(None))
            .all()
        )

        geocoded_count = 0

        for location in locations:
            try:
                lat, lon = await cls.geocode_location(location.address)
                location.latitude = lat
                location.longitude = lon
                location.geocoded_at = datetime.now(UTC)
                geocoded_count += 1
            except GeocodeError as e:
                logger.warning(f"Failed to geocode {location.location_id}: {e.message}")
                continue

        if geocoded_count > 0:
            db.commit()
            logger.info(f"Successfully geocoded {geocoded_count} locations")

        return geocoded_count
