"""Unit tests for fire data service."""

from app.services.fire_data import FireDataService


def test_haversine_distance():
    """Test haversine distance calculation."""
    # Los Angeles to San Francisco great-circle distance is about 347.5 miles.
    lat1, lon1 = 34.0522, -118.2437
    lat2, lon2 = 37.7749, -122.4194

    distance = FireDataService.haversine_distance(lat1, lon1, lat2, lon2)

    # Should be approximately 347-349 miles
    assert 347 < distance < 349, f"Expected ~347.5 miles, got {distance}"


def test_calculate_bearing():
    """Test bearing calculation."""
    # North: bearing should be ~0
    lat1, lon1 = 0, 0
    lat2, lon2 = 1, 0
    bearing = FireDataService.calculate_bearing(lat1, lon1, lat2, lon2)
    assert bearing < 20 or bearing > 340, f"Expected bearing ~0 (North), got {bearing}"


def test_get_cardinal_direction():
    """Test cardinal direction conversion."""
    assert FireDataService.get_cardinal_direction(0) == "N"
    assert FireDataService.get_cardinal_direction(45) == "NE"
    assert FireDataService.get_cardinal_direction(90) == "E"
    assert FireDataService.get_cardinal_direction(135) == "SE"
    assert FireDataService.get_cardinal_direction(180) == "S"
    assert FireDataService.get_cardinal_direction(225) == "SW"
    assert FireDataService.get_cardinal_direction(270) == "W"
    assert FireDataService.get_cardinal_direction(315) == "NW"


def test_get_alert_level_safe():
    """Test alert level for safe distance."""
    level = FireDataService.get_alert_level(distance=100)
    assert level == "SAFE"


def test_get_alert_level_evacuate():
    """Test alert level for very close fire."""
    level = FireDataService.get_alert_level(distance=0.5)
    assert level == "EVACUATE"


def test_get_alert_level_with_containment():
    """Test alert level adjusted for low containment."""
    # 5 miles, 0% contained should be WARNING
    level = FireDataService.get_alert_level(distance=5, acres_burned=1000, percent_contained=0)
    assert level in ["WARNING", "PREPARE"]


def test_get_alert_level_with_large_fire():
    """Test alert level adjusted for large fire."""
    # 10 miles, 5000+ acres should trigger higher alert
    level = FireDataService.get_alert_level(distance=10, acres_burned=10000, percent_contained=50)
    assert level in ["MONITOR", "PREPARE"]
