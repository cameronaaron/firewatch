"""GeoJSON allows "properties": null; it must read as no properties."""

import asyncio

from app.core.config import get_settings
from app.services.external_intelligence import ExternalIntelligenceService


def test_road_closure_with_null_properties(monkeypatch):
    service = ExternalIntelligenceService()
    monkeypatch.setattr(
        get_settings(), "ROAD_CLOSURES_GEOJSON_URL", "https://closures.example/feed.geojson"
    )

    async def features(url):
        return [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-118.0, 34.0]},
                "properties": None,
            }
        ]

    monkeypatch.setattr(service, "_fetch_geojson_features", features)
    summary = asyncio.run(service.fetch_road_closures(34.0, -118.0))
    assert summary["closure_count_nearby"] == 1
    assert summary["nearest_closure"]["road_name"] is None
