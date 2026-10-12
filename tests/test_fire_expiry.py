"""Fires unreported for FIRE_STALE_HOURS retire; a reported fire is active again."""

from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.models.database import Fire
from app.services.fire_data import FireDataService


def _fire(fire_id, seen_hours_ago, now, active=True):
    return Fire(
        fire_id=fire_id,
        name=fire_id,
        latitude=34.0,
        longitude=-118.0,
        is_active=active,
        last_updated=now - timedelta(hours=seen_hours_ago),
    )


def test_fires_unseen_past_the_window_retire(db, monkeypatch):
    monkeypatch.setattr(get_settings(), "FIRE_STALE_HOURS", 6.0)
    now = datetime.now(UTC)
    db.add_all([_fire("recent", 1, now), _fire("stale", 7, now)])
    db.commit()

    assert FireDataService.retire_stale_fires(db, now) == 1
    db.commit()
    active = {f.fire_id for f in db.query(Fire).filter(Fire.is_active)}
    assert active == {"recent"}


def test_a_reported_fire_is_reactivated(db):
    now = datetime.now(UTC)
    db.add(_fire("back", 30, now, active=False))
    db.commit()

    FireDataService.upsert_fire(
        db,
        {
            "fire_id": "back",
            "name": "Back",
            "latitude": 34.0,
            "longitude": -118.0,
            "acres_burned": 10.0,
            "percent_contained": 0.0,
        },
    )
    db.commit()
    fire = db.query(Fire).filter(Fire.fire_id == "back").one()
    assert fire.is_active is True
    assert FireDataService.retire_stale_fires(db, datetime.now(UTC)) == 0
