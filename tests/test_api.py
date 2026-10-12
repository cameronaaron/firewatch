"""Unit tests for API endpoints."""

import pytest

from app.api import locations as locations_api
from app.models.database import Location


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_status_endpoint(client):
    """Test status endpoint."""
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert "version" in data


def test_data_source_inventory_endpoint(client):
    """Test the code-backed wildfire data-source inventory endpoint."""
    response = client.get("/api/v1/predictions/data-sources")
    assert response.status_code == 200

    data = response.json()
    assert "overall_assessment" in data
    assert "implemented_now" in data
    assert "highest_impact_gaps" in data
    assert data["recommended_priority_order"]
    assert any(
        feed["name"] == "CAL FIRE incidents" for feed in data["implemented_now"]["incident_feeds"]
    )
    assert any(
        gap["name"] == "Historical training truth tied to incidents, perimeters, weather, and fuels"
        for gap in data["highest_impact_gaps"]
    )


def test_list_locations_empty(client):
    """Test listing locations when empty."""
    response = client.get("/api/v1/locations")
    assert response.status_code == 200
    assert response.json() == []


def test_create_location(client, db, monkeypatch):
    """Test creating a location."""

    async def fake_geocode(address):
        return 34.0, -118.0

    monkeypatch.setattr(
        locations_api.GeocodingService, "geocode_location", staticmethod(fake_geocode)
    )

    location_data = {
        "location_id": "test_house",
        "address": "123 Test St, Test City, CA",
        "display_name": "Test House",
        "icon": "🏠",
    }

    response = client.post("/api/v1/locations", json=location_data)
    assert response.status_code == 200
    data = response.json()
    assert data["location_id"] == "test_house"
    assert data["display_name"] == "Test House"
    assert data["address"] is None
    assert data["latitude"] == 34.0


def test_create_location_with_coordinates_only(client, db):
    """Test creating a location without persisting the submitted address."""
    location_data = {
        "location_id": "coord_house",
        "display_name": "Coordinate House",
        "icon": "📍",
        "latitude": 34.1,
        "longitude": -118.2,
    }

    response = client.post("/api/v1/locations", json=location_data)
    assert response.status_code == 200
    data = response.json()
    assert data["location_id"] == "coord_house"
    assert data["address"] is None
    assert data["latitude"] == 34.1
    assert data["longitude"] == -118.2


def test_resolve_location_preview(client, monkeypatch):
    """Test preview geocoding without persisting the submitted address."""

    async def fake_geocode(address):
        return 35.0, -117.0

    monkeypatch.setattr(
        locations_api.GeocodingService, "geocode_location", staticmethod(fake_geocode)
    )

    response = client.post(
        "/api/v1/locations/resolve", json={"address": "123 Preview Way, Sampletown, CA"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["latitude"] == 35.0
    assert data["longitude"] == -117.0


def test_resolve_location_preview_with_zipcode(client, monkeypatch):
    """Test ZIP-only preview requests remain valid inputs for address resolution."""
    seen = {}

    async def fake_geocode(address):
        seen["address"] = address
        return 34.0901, -118.4065

    monkeypatch.setattr(
        locations_api.GeocodingService, "geocode_location", staticmethod(fake_geocode)
    )

    response = client.post("/api/v1/locations/resolve", json={"address": "90210"})
    assert response.status_code == 200
    data = response.json()
    assert seen["address"] == "90210"
    assert data["latitude"] == pytest.approx(34.0901)
    assert data["longitude"] == pytest.approx(-118.4065)


def test_get_location(client, db):
    """Test getting a specific location."""
    # Create location
    location = Location(
        location_id="test_house",
        address="123 Test St",
        display_name="Test House",
        icon="🏠",
        latitude=34.0,
        longitude=-118.0,
    )
    db.add(location)
    db.commit()

    # Get location
    response = client.get("/api/v1/locations/test_house")
    assert response.status_code == 200
    data = response.json()
    assert data["location_id"] == "test_house"


def test_get_nonexistent_location(client):
    """Test getting non-existent location."""
    response = client.get("/api/v1/locations/nonexistent")
    assert response.status_code == 404


def test_list_fires_empty(client):
    """Test listing fires when empty."""
    response = client.get("/api/v1/fires")
    assert response.status_code == 200
    assert response.json() == []


def test_create_duplicate_location(client, db):
    """Test creating duplicate location."""
    location1 = Location(
        location_id="test_house", address="123 Test St", display_name="Test House", icon="🏠"
    )
    db.add(location1)
    db.commit()

    location_data = {
        "location_id": "test_house",
        "address": "456 Other St",
        "display_name": "Other House",
    }

    response = client.post("/api/v1/locations", json=location_data)
    assert response.status_code == 409  # Conflict


def test_threat_analysis_ignores_inactive_fires(client, db):
    from app.models.database import Fire, Location

    db.add(
        Location(
            location_id="loc-x",
            address="1 Test Way",
            display_name="X",
            latitude=34.0,
            longitude=-118.0,
        )
    )
    db.add(Fire(fire_id="out-1", name="Out", latitude=34.01, longitude=-118.0, is_active=False))
    db.commit()

    body = client.get("/api/v1/predictions/threat/loc-x").json()
    assert body["status"] == "safe"
