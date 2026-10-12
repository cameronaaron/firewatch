import pytest

from app.ml.fire_weather import danger_level, equilibrium_moisture, fosberg_index


def test_red_flag_conditions_rate_high_and_damp_calm_ones_low():
    # NWS red-flag territory: hot, RH near 10%, sustained 20 mph wind.
    assert danger_level(fosberg_index(100, 10, 20)) in {"HIGH", "VERY_HIGH", "EXTREME"}
    assert danger_level(fosberg_index(60, 80, 0)) == "LOW"


def test_index_rises_with_wind_and_heat_and_falls_with_humidity():
    assert fosberg_index(90, 20, 5) < fosberg_index(90, 20, 15) < fosberg_index(90, 20, 25)
    assert fosberg_index(70, 20, 10) < fosberg_index(100, 20, 10)
    assert fosberg_index(90, 60, 10) < fosberg_index(90, 30, 10) < fosberg_index(90, 5, 10)


def test_index_is_bounded_and_reaches_its_ceiling():
    for temp in (-20, 40, 80, 120):
        for rh in (-5, 0, 9.9, 10, 49.9, 50, 100, 120):
            for wind in (-3, 0, 10, 60):
                assert 0 <= fosberg_index(temp, rh, wind) <= 100
    # Fosberg's scale: bone-dry fuel in a 30 mph wind is ~100.
    assert fosberg_index(70, 0, 30) == pytest.approx(100, abs=0.5)
    # Fuel moisture never passes extinction (30%), and saturated air keeps
    # even a strong wind LOW.
    assert all(equilibrium_moisture(t, h) <= 30 for t in (-40, 0, 40) for h in (60, 100, 150))
    assert danger_level(fosberg_index(40, 100, 30)) == "LOW"


def test_danger_bands():
    assert [danger_level(x) for x in (0, 19.9, 20, 39.9, 40, 60, 80, 100)] == [
        "LOW",
        "LOW",
        "MODERATE",
        "MODERATE",
        "HIGH",
        "VERY_HIGH",
        "EXTREME",
        "EXTREME",
    ]
