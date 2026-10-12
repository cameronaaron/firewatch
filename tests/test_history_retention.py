from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.services.fire_data import FireDataService


def test_movement_history_honours_the_retention_setting(monkeypatch):
    monkeypatch.setattr(get_settings(), "HISTORICAL_DATA_RETENTION_HOURS", 2)
    key = "fire-r_loc-r"
    now = datetime.now(UTC)
    monkeypatch.setitem(
        FireDataService._distance_history_cache,
        key,
        [
            {"timestamp": now - timedelta(hours=3), "distance": 9.0},
            {"timestamp": now - timedelta(hours=1), "distance": 8.0},
        ],
    )

    FireDataService.calculate_movement("fire-r", "loc-r", 7.0)

    kept = FireDataService._distance_history_cache[key]
    assert [entry["distance"] for entry in kept] == [8.0, 7.0]
