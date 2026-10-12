"""A cancelled FIRMS fetch is skipped like a failed one, not iterated."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from app.services.fire_data import FireDataService
from app.services.satellite_service import FireDetection, SatelliteFireService


def _detection():
    return FireDetection(
        latitude=34.0,
        longitude=-118.0,
        confidence=90,
        sensor="VIIRS",
        timestamp=datetime(2026, 5, 21, tzinfo=UTC),
    )


def test_satellite_skips_a_cancelled_source(monkeypatch):
    service = SatelliteFireService()
    service.firms_map_key = "test-key"
    service.firms_sources = ("A", "B")

    async def fetch(source, bbox, days):
        if source == "A":
            raise asyncio.CancelledError
        return [_detection()]

    monkeypatch.setattr(service, "_fetch_firms_area_source", fetch)
    detections = asyncio.run(service.get_firms_detections(34.0, -118.0))
    assert len(detections) == 1


def test_fire_data_skips_a_cancelled_location(monkeypatch):
    calls = []

    class Satellite:
        async def get_firms_detections(self, latitude, longitude):
            calls.append(latitude)
            if latitude == 1.0:
                raise asyncio.CancelledError
            return [_detection()]

    monkeypatch.setattr(
        FireDataService, "_get_satellite_service", classmethod(lambda cls: Satellite())
    )
    locations = [
        SimpleNamespace(latitude=1.0, longitude=1.0),
        SimpleNamespace(latitude=34.0, longitude=-118.0),
    ]
    hotspots = asyncio.run(FireDataService._fetch_firms_near_locations(locations))
    assert sorted(calls) == [1.0, 34.0]
    assert isinstance(hotspots, list)
