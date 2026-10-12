"""API routes for alerts and threat assessment."""

import logging
from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.database import Alert, DistanceHistory, Fire, Location
from app.schemas import AlertLevelEnum, AlertResponse, ThreatAssessment
from app.services import AirQualityService, FireDataService, SmokeDataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])
smoke_service = SmokeDataService()
air_quality_service = AirQualityService()
settings = get_settings()
_ALERT_SEVERITY = {
    "UNKNOWN": 0,
    "SAFE": 1,
    "MONITOR": 2,
    "PREPARE": 3,
    "WARNING": 4,
    "EVACUATE": 5,
}


@router.get("", response_model=list[AlertResponse])
def list_alerts(
    location_id: str = Query(None),
    alert_level: AlertLevelEnum = Query(None),
    acknowledged: bool = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Get alerts with optional filtering."""
    query = db.query(Alert)

    if location_id:
        location = db.query(Location).filter(Location.location_id == location_id).first()
        if not location:
            raise HTTPException(status_code=404, detail="Location not found")
        query = query.filter(Alert.location_id == location.id)

    if alert_level:
        query = query.filter(Alert.alert_level == alert_level)

    if acknowledged is not None:
        query = query.filter(Alert.acknowledged == acknowledged)

    alerts = query.order_by(Alert.timestamp.desc()).limit(limit).all()
    return alerts


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    """Get specific alert."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.put("/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    """Mark alert as acknowledged."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.acknowledged = True
    alert.acknowledged_at = datetime.now(UTC)
    db.commit()

    return {"message": "Alert acknowledged"}


@router.get("/assessment/{location_id}", response_model=ThreatAssessment)
def get_threat_assessment(location_id: str, db: Session = Depends(get_db)):
    """Get threat assessment for a location."""
    # Get location
    location = db.query(Location).filter(Location.location_id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    if location.latitude is None or location.longitude is None:
        raise HTTPException(status_code=400, detail="Location not geocoded")

    smoke_assessment = smoke_service.get_smoke_assessment(location.latitude, location.longitude)
    air_quality_assessment = air_quality_service.get_air_quality_assessment(
        location.latitude,
        location.longitude,
    )

    # Get most recent distance history entry for this location
    latest = (
        db.query(DistanceHistory)
        .filter(DistanceHistory.location_id == location.id)
        .order_by(DistanceHistory.timestamp.desc())
        .first()
    )

    if not latest:
        live_snapshot = _get_live_fire_snapshot(location, db)
        if live_snapshot:
            fire, fallback = live_snapshot
            fire_alert_level = FireDataService.get_alert_level(
                fallback.distance_miles,
                fire.acres_burned,
                fire.percent_contained,
            )
            smoke_alert_level = _get_smoke_alert_level(smoke_assessment)
            air_quality_alert_level = _get_air_quality_alert_level(air_quality_assessment)
            alert_level = _combine_alert_levels(
                fire_alert_level,
                smoke_alert_level,
                air_quality_alert_level,
            )
            recommendations = _get_recommendations(
                fire_alert_level,
                fire,
                fallback,
                smoke_assessment,
                air_quality_assessment,
            )

            return ThreatAssessment(
                location_id=location.location_id,
                location_display_name=location.display_name,
                alert_level=AlertLevelEnum[alert_level],
                nearest_fire_name=fire.name,
                distance_miles=fallback.distance_miles,
                direction=fallback.direction,
                acres_burned=fire.acres_burned,
                percent_contained=fire.percent_contained,
                moving_towards=False,
                speed_mph=0.0,
                smoke_density=smoke_assessment.get("smoke_density") if smoke_assessment else None,
                smoke_distance_miles=smoke_assessment.get("smoke_distance_miles")
                if smoke_assessment
                else None,
                smoke_source=smoke_assessment.get("smoke_source") if smoke_assessment else None,
                smoke_observed_at=smoke_assessment.get("smoke_observed_at")
                if smoke_assessment
                else None,
                air_quality_index=air_quality_assessment.get("air_quality_index")
                if air_quality_assessment
                else None,
                pm25_ug_m3=air_quality_assessment.get("pm25_ug_m3")
                if air_quality_assessment
                else None,
                air_quality_category=air_quality_assessment.get("air_quality_category")
                if air_quality_assessment
                else None,
                air_quality_source=air_quality_assessment.get("air_quality_source")
                if air_quality_assessment
                else None,
                air_quality_observed_at=air_quality_assessment.get("air_quality_observed_at")
                if air_quality_assessment
                else None,
                recommendations=recommendations,
            )

        smoke_alert_level = _get_smoke_alert_level(smoke_assessment)
        air_quality_alert_level = _get_air_quality_alert_level(air_quality_assessment)
        return ThreatAssessment(
            location_id=location.location_id,
            location_display_name=location.display_name,
            alert_level=AlertLevelEnum[
                _combine_alert_levels(smoke_alert_level, air_quality_alert_level)
            ],
            nearest_fire_name=None,
            distance_miles=None,
            direction=None,
            acres_burned=None,
            percent_contained=None,
            moving_towards=False,
            speed_mph=0.0,
            smoke_density=smoke_assessment.get("smoke_density") if smoke_assessment else None,
            smoke_distance_miles=smoke_assessment.get("smoke_distance_miles")
            if smoke_assessment
            else None,
            smoke_source=smoke_assessment.get("smoke_source") if smoke_assessment else None,
            smoke_observed_at=smoke_assessment.get("smoke_observed_at")
            if smoke_assessment
            else None,
            air_quality_index=air_quality_assessment.get("air_quality_index")
            if air_quality_assessment
            else None,
            pm25_ug_m3=air_quality_assessment.get("pm25_ug_m3") if air_quality_assessment else None,
            air_quality_category=air_quality_assessment.get("air_quality_category")
            if air_quality_assessment
            else None,
            air_quality_source=air_quality_assessment.get("air_quality_source")
            if air_quality_assessment
            else None,
            air_quality_observed_at=air_quality_assessment.get("air_quality_observed_at")
            if air_quality_assessment
            else None,
            recommendations=_get_no_fire_recommendations(smoke_assessment, air_quality_assessment),
        )

    # Get fire details
    fire = latest.fire

    # Calculate alert level
    fire_alert_level = FireDataService.get_alert_level(
        latest.distance_miles, fire.acres_burned, fire.percent_contained
    )
    smoke_alert_level = _get_smoke_alert_level(smoke_assessment)
    air_quality_alert_level = _get_air_quality_alert_level(air_quality_assessment)
    alert_level = _combine_alert_levels(
        fire_alert_level, smoke_alert_level, air_quality_alert_level
    )

    # Generate recommendations
    recommendations = _get_recommendations(
        fire_alert_level,
        fire,
        latest,
        smoke_assessment,
        air_quality_assessment,
    )

    return ThreatAssessment(
        location_id=location.location_id,
        location_display_name=location.display_name,
        alert_level=AlertLevelEnum[alert_level],
        nearest_fire_name=fire.name,
        distance_miles=latest.distance_miles,
        direction=latest.direction,
        acres_burned=fire.acres_burned,
        percent_contained=fire.percent_contained,
        moving_towards=latest.moving_towards,
        speed_mph=latest.speed_mph,
        smoke_density=smoke_assessment.get("smoke_density") if smoke_assessment else None,
        smoke_distance_miles=smoke_assessment.get("smoke_distance_miles")
        if smoke_assessment
        else None,
        smoke_source=smoke_assessment.get("smoke_source") if smoke_assessment else None,
        smoke_observed_at=smoke_assessment.get("smoke_observed_at") if smoke_assessment else None,
        air_quality_index=air_quality_assessment.get("air_quality_index")
        if air_quality_assessment
        else None,
        pm25_ug_m3=air_quality_assessment.get("pm25_ug_m3") if air_quality_assessment else None,
        air_quality_category=air_quality_assessment.get("air_quality_category")
        if air_quality_assessment
        else None,
        air_quality_source=air_quality_assessment.get("air_quality_source")
        if air_quality_assessment
        else None,
        air_quality_observed_at=air_quality_assessment.get("air_quality_observed_at")
        if air_quality_assessment
        else None,
        recommendations=recommendations,
    )


def _get_live_fire_snapshot(location: Location, db: Session):
    """Compute nearest active fire directly when distance history has not been populated yet."""
    nearest_fire: Fire | None = None
    nearest_distance: float | None = None
    nearest_direction: str | None = None

    coords = location.coordinates
    if coords is None:
        return None
    fires = db.query(Fire).filter(Fire.is_active).all()
    for fire in fires:
        distance = FireDataService.haversine_distance(fire.latitude, fire.longitude, *coords)
        if distance > settings.MAX_FIRE_DISTANCE_MILES:
            continue

        if nearest_distance is None or distance < nearest_distance:
            bearing = FireDataService.calculate_bearing(fire.latitude, fire.longitude, *coords)
            nearest_distance = round(distance, 1)
            nearest_direction = (
                f"{FireDataService.get_cardinal_direction(bearing)} ({int(bearing)}°)"
            )
            nearest_fire = fire

    if nearest_fire is None or nearest_distance is None or nearest_direction is None:
        return None

    history_like = SimpleNamespace(
        distance_miles=nearest_distance,
        direction=nearest_direction,
        moving_towards=False,
        speed_mph=0.0,
    )
    return nearest_fire, history_like


def _get_smoke_alert_level(smoke_assessment: dict | None) -> str:
    """Translate a smoke assessment into a low-grade alert signal."""
    if not smoke_assessment:
        return "SAFE"
    return "MONITOR"


def _get_air_quality_alert_level(air_quality_assessment: dict | None) -> str:
    """Translate AQI into a low-grade alert signal that complements fire proximity."""
    if not air_quality_assessment:
        return "SAFE"

    aqi = air_quality_assessment.get("air_quality_index")
    if aqi is None:
        return "SAFE"
    if aqi >= 151:
        return "WARNING"
    if aqi >= 101:
        return "PREPARE"
    if aqi >= 51:
        return "MONITOR"
    return "SAFE"


def _combine_alert_levels(*levels: str) -> str:
    """Return the highest severity alert level across multiple signals."""
    return max(levels, key=lambda level: _ALERT_SEVERITY.get(level, 0))


def _get_no_fire_recommendations(
    smoke_assessment: dict | None,
    air_quality_assessment: dict | None,
) -> list[str]:
    """Return recommendations when smoke is the only current signal."""
    if not smoke_assessment and not air_quality_assessment:
        return [
            "No active fires detected nearby",
            "Stay informed of weather conditions",
            "Review evacuation plan regularly",
        ]

    recommendations: list[str] = ["No active fires detected nearby"]

    if smoke_assessment:
        density = smoke_assessment.get("smoke_density", "Smoke")
        scope = smoke_assessment.get("smoke_scope")
        distance = smoke_assessment.get("smoke_distance_miles")
        recommendations.append(
            f"{density} smoke is currently over your location"
            if scope == "overhead"
            else f"{density} smoke is nearby at about {distance} miles"
        )

    if air_quality_assessment:
        recommendations.append(_format_air_quality_summary(air_quality_assessment))

    recommendations.append("Limit outdoor exposure if smoke or pollution is affecting your area")
    recommendations.append("Close windows and monitor local air quality updates")
    return recommendations


def _get_recommendations(
    alert_level: str,
    fire: Fire,
    history: DistanceHistory | SimpleNamespace,
    smoke_assessment: dict | None = None,
    air_quality_assessment: dict | None = None,
) -> list[str]:
    """Generate recommended actions based on alert level."""
    recommendations = {
        "SAFE": [
            "Stay informed of fire conditions",
            "Review your evacuation plan",
            "Monitor local news and alerts",
        ],
        "MONITOR": [
            "Pack essential items",
            "Prepare important documents",
            "Keep phones charged and ready",
            f"Monitor {fire.name} fire movement",
        ],
        "PREPARE": [
            "⚠️ ALERT: Fire approaching your area",
            "Pack vehicles with emergency supplies",
            "Prepare pets for possible evacuation",
            "Take photos of property for insurance",
            f"Fire is {history.distance_miles:.1f} miles {history.direction}",
        ],
        "WARNING": [
            "🔥 URGENT: PREPARE TO EVACUATE",
            f"{fire.name} fire is {history.distance_miles:.1f} miles {history.direction}",
            f"Fire has burned {int(fire.acres_burned)} acres",
            "Gather family members and pets",
            "Follow evacuation routes when ordered",
        ],
        "EVACUATE": [
            "🚨 EVACUATE IMMEDIATELY",
            f"Fire is {history.distance_miles:.1f} miles {history.direction}",
            "Leave now if you haven't already",
            "Follow emergency responder instructions",
            "Call 911 if you need assistance",
        ],
        "UNKNOWN": ["Data unknown; remain on alert"],
    }

    selected = list(recommendations.get(alert_level, ["Stay informed"]))
    if smoke_assessment:
        density = smoke_assessment.get("smoke_density", "Smoke")
        scope = smoke_assessment.get("smoke_scope")
        distance = smoke_assessment.get("smoke_distance_miles")
        selected.append(
            f"{density} smoke is currently over your location"
            if scope == "overhead"
            else f"{density} smoke is nearby at about {distance} miles"
        )
        selected.append("Limit outdoor exposure and monitor local air quality conditions")

    if air_quality_assessment:
        selected.append(_format_air_quality_summary(air_quality_assessment))
        selected.append(
            "Use indoor air filtration if AQI is elevated and avoid strenuous outdoor activity"
        )

    return selected


def _format_air_quality_summary(air_quality_assessment: dict) -> str:
    """Return a compact human-readable air-quality status line."""
    category = air_quality_assessment.get("air_quality_category") or "Unknown"
    aqi = air_quality_assessment.get("air_quality_index")
    pm25 = air_quality_assessment.get("pm25_ug_m3")
    details = []
    if aqi is not None:
        details.append(f"US AQI {aqi}")
    if pm25 is not None:
        details.append(f"PM2.5 {pm25:.1f} ug/m3")
    detail_text = f" ({', '.join(details)})" if details else ""
    return f"Current air quality is {category}{detail_text}"
