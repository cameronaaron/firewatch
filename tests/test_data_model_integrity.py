"""Data-model integrity tests for ORM constraints and defaults."""

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.database import Alert, DistanceHistory, Fire, Location


def _create_location(db, *, location_id: str = "loc-model") -> Location:
    location = Location(
        location_id=location_id,
        address="123 Model St",
        display_name="Model Location",
        icon="home",
        latitude=34.0,
        longitude=-118.0,
    )
    db.add(location)
    db.commit()
    db.refresh(location)
    return location


def _create_fire(db, *, fire_id: str = "fire-model", percent_contained: float = 20.0) -> Fire:
    fire = Fire(
        fire_id=fire_id,
        name="Model Fire",
        latitude=34.2,
        longitude=-118.2,
        acres_burned=1200.0,
        percent_contained=percent_contained,
    )
    db.add(fire)
    db.commit()
    db.refresh(fire)
    return fire


def test_fire_extra_metadata_uses_callable_default():
    default_arg = Fire.__table__.c.extra_metadata.default.arg
    assert callable(default_arg)
    assert not isinstance(default_arg, dict)


def test_fire_percent_contained_constraint_rejects_out_of_range(db):
    db.add(
        Fire(
            fire_id="fire-invalid-contained",
            name="Invalid Fire",
            latitude=34.1,
            longitude=-118.1,
            acres_burned=100.0,
            percent_contained=150.0,
        )
    )

    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_location_coordinate_pair_constraint_rejects_partial_coordinates(db):
    db.add(
        Location(
            location_id="loc-partial-coords",
            address="Unknown",
            display_name="Bad Coords",
            icon="home",
            latitude=34.0,
            longitude=None,
        )
    )

    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_distance_history_constraint_rejects_negative_distance(db):
    location = _create_location(db)
    fire = _create_fire(db)

    db.add(
        DistanceHistory(
            fire_id=fire.id,
            location_id=location.id,
            distance_miles=-1.0,
            bearing_degrees=90.0,
            direction="E",
            speed_mph=0.4,
        )
    )

    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_alert_level_constraint_rejects_invalid_values(db):
    location = _create_location(db, location_id="loc-alert")
    fire = _create_fire(db, fire_id="fire-alert")

    db.add(
        Alert(
            fire_id=fire.id,
            location_id=location.id,
            alert_level="PANIC",
            distance_miles=1.5,
            reason="Invalid level test",
        )
    )

    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_location_constraint_rejects_blank_display_name(db):
    db.add(
        Location(
            location_id="loc-blank-name",
            address="123 Test Ave",
            display_name="   ",
            icon="home",
            latitude=34.0,
            longitude=-118.0,
        )
    )

    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_fire_constraint_rejects_blank_name(db):
    db.add(
        Fire(
            fire_id="fire-blank-name",
            name="   ",
            latitude=34.1,
            longitude=-118.1,
            acres_burned=100.0,
            percent_contained=50.0,
        )
    )

    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
