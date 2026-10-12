"""Fosberg Fire Weather Index (Fosberg 1978).

Rates how fast fire could spread from one weather observation: temperature,
relative humidity and wind. The Canadian FWI system needs a running record of
prior days, which this app does not keep; Fosberg needs none.
"""

import math

# This app's bands, not a published standard.
DANGER_BANDS = ((20.0, "LOW"), (40.0, "MODERATE"), (60.0, "HIGH"), (80.0, "VERY_HIGH"))


def equilibrium_moisture(temp_f: float, rh_pct: float) -> float:
    """Fine dead fuel moisture in percent (Simard 1968), capped at extinction (30%)."""
    h = min(100.0, max(0.0, rh_pct))
    t = temp_f
    if h < 10:
        m = 0.03229 + 0.281073 * h - 0.000578 * h * t
    elif h < 50:
        m = 2.22749 + 0.160107 * h - 0.01478 * t
    else:
        m = 21.0606 + 0.005565 * h * h - 0.00035 * h * t - 0.483199 * h
    return min(30.0, max(0.0, m))


def fosberg_index(temp_f: float, rh_pct: float, wind_mph: float) -> float:
    """Fosberg index on 0-100; 100 is bone-dry fuel in a 30 mph wind."""
    r = equilibrium_moisture(temp_f, rh_pct) / 30.0
    damping = 1 - 2 * r + 1.5 * r**2 - 0.5 * r**3
    wind = max(0.0, wind_mph)
    return min(100.0, max(0.0, damping * math.sqrt(1 + wind * wind) / 0.3002))


def danger_level(index: float) -> str:
    for upper, label in DANGER_BANDS:
        if index < upper:
            return label
    return "EXTREME"
