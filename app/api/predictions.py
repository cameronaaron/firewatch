"""Prediction endpoints: threat, evacuation, forecast, spread, anomalies and risk."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.database import Fire, Location
from app.services.comprehensive_data import ComprehensiveDataService
from app.services.fire_data import FireDataService
from app.services.ml_analysis import MLFireAnalysisService, distance_history

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])

# Initialize ML analysis service
ml_service = MLFireAnalysisService()


def _distance_between_fire_and_location(fire: Fire, location: Location) -> float:
    """Calculate fire-to-location distance in miles."""
    if (
        getattr(fire, "latitude", None) is None
        or getattr(fire, "longitude", None) is None
        or getattr(location, "latitude", None) is None
        or getattr(location, "longitude", None) is None
    ):
        return 0.0

    return FireDataService.haversine_distance(
        fire.latitude,
        fire.longitude,
        location.latitude,
        location.longitude,
    )


@router.get("/data-sources")
def get_data_source_inventory():
    """List the wildfire data sources in use and the most valuable ones still missing."""
    return ComprehensiveDataService.build_data_source_inventory()


@router.get("/threat/{location_id}")
async def get_threat_prediction(
    location_id: str,
    db: Session = Depends(get_db),
):
    """
    Get comprehensive ML-based threat prediction for a location.

    - Threat score (0-100)
    - Fire spread predictions
    - Direction probabilities
    - Weather impact analysis
    - Recommendations
    - Anomaly detection
    """
    location = db.query(Location).filter(Location.location_id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    if location.latitude is None or location.longitude is None:
        raise HTTPException(status_code=400, detail="Location not geocoded")

    fires = db.query(Fire).filter(Fire.is_active).all()

    if not fires:
        return {
            "location_id": location_id,
            "status": "safe",
            "message": "No active fires detected",
            "threat_score": 0,
        }

    distance_pairs = sorted(
        ((_distance_between_fire_and_location(fire, location), fire) for fire in fires),
        key=lambda pair: pair[0],
    )
    nearest_distance, nearest_fire = distance_pairs[0]

    threat_analysis = await ml_service.analyze_fire_threat(
        nearest_fire,
        location,
        nearest_distance,
        db,
    )

    return {
        "location_id": location_id,
        "analysis_timestamp": None,
        **threat_analysis,
    }


@router.get("/evacuation/{location_id}")
async def get_evacuation_plan(
    location_id: str,
    db: Session = Depends(get_db),
):
    """
    Get optimized evacuation plan with prioritized routes.

    Considers:
    - Multiple nearby fires
    - Wind patterns
    - Road conditions
    - Timing recommendations
    """
    location = db.query(Location).filter(Location.location_id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    if location.latitude is None or location.longitude is None:
        raise HTTPException(status_code=400, detail="Location not geocoded")

    fires = db.query(Fire).filter(Fire.is_active).all()

    evacuation_plan = await ml_service.get_evacuation_optimization(
        location,
        fires,
        db,
    )

    return {
        "location_id": location_id,
        **evacuation_plan,
    }


@router.get("/forecast/{location_id}")
async def get_predictive_forecast(
    location_id: str,
    hours: int = Query(24, ge=1, le=336),
    db: Session = Depends(get_db),
):
    """
    Get predictive analytics for upcoming period.

    Includes:
    - Weather forecast
    - Fire weather indices
    - Risk trends
    - Spread multipliers

    Args:
        location_id: Location identifier
        hours: Forecast period in hours (1-336, default 24)
    """
    location = db.query(Location).filter(Location.location_id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    if location.latitude is None or location.longitude is None:
        raise HTTPException(status_code=400, detail="Location not geocoded")

    forecast = await ml_service.get_predictive_analytics(
        location,
        forecast_hours=hours,
    )

    return forecast


@router.get("/spread-rate/{fire_id}")
def get_fire_spread_rate_prediction(
    fire_id: str,
    location_id: str = Query(None),
    db: Session = Depends(get_db),
):
    """
    Get predicted fire spread rate.

    Returns:
        Spread rate prediction in miles per hour
    """
    fire = db.query(Fire).filter(Fire.fire_id == fire_id).first()
    if not fire:
        raise HTTPException(status_code=404, detail="Fire not found")

    history_location = None
    normalized_location_id = location_id if isinstance(location_id, str) else None
    if normalized_location_id:
        history_location = (
            db.query(Location).filter(Location.location_id == normalized_location_id).first()
        )

    history = distance_history(db, fire, history_location)

    predictor = ml_service.predictor
    spread_rate = predictor.predict_spread_rate(fire, history)

    # Get weather impact
    weather_multiplier = 1.0
    if hasattr(fire, "latitude"):
        return {
            "fire_id": fire_id,
            "predicted_spread_rate_mph": float(spread_rate),
            "weather_multiplier": float(weather_multiplier),
            "adjusted_spread_rate_mph": float(spread_rate * weather_multiplier),
            "confidence": 0.7,
            "model_provenance": predictor.get_model_provenance(),
        }

    return {
        "fire_id": fire_id,
        "predicted_spread_rate_mph": float(spread_rate),
        "weather_multiplier": float(weather_multiplier),
        "adjusted_spread_rate_mph": float(spread_rate * weather_multiplier),
        "confidence": 0.5,
        "model_provenance": predictor.get_model_provenance(),
    }


@router.get("/anomalies/{fire_id}")
def detect_fire_anomalies(
    fire_id: str,
    db: Session = Depends(get_db),
):
    """
    Detect anomalous fire behavior patterns.

    Identifies:
    - Unusual spread rates
    - Erratic direction changes
    - Unexpected acceleration
    - Statistical outliers
    """
    fire = db.query(Fire).filter(Fire.fire_id == fire_id).first()
    if not fire:
        raise HTTPException(status_code=404, detail="Fire not found")

    history = distance_history(db, fire)

    detector = ml_service.anomaly_detector
    result = detector.detect_anomaly(fire, history)

    return {
        "fire_id": fire_id,
        "is_anomalous": result.get("is_anomaly"),
        "anomaly_score": result.get("anomaly_score"),
        "reasons": result.get("reasons", []),
        "insight": detector.get_anomaly_insight(fire, history),
    }


@router.get("/impact-timeline/{fire_id}/{location_id}")
def get_fire_impact_timeline(
    fire_id: str,
    location_id: str,
    target_distance: float = Query(0.5, ge=0.1, le=50),
    db: Session = Depends(get_db),
):
    """
    Get predicted timeline for fire impact.

    Predicts when fire will reach specific distance from location.

    Args:
        fire_id: Fire identifier
        location_id: Location identifier
        target_distance: Distance to predict arrival at (miles)
    """
    fire = db.query(Fire).filter(Fire.fire_id == fire_id).first()
    if not fire:
        raise HTTPException(status_code=404, detail="Fire not found")

    location = db.query(Location).filter(Location.location_id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    history = distance_history(db, fire, location)

    predictor = ml_service.predictor
    timeline = predictor.predict_impact_timeline(fire, history, target_distance)

    return {
        "fire_id": fire_id,
        "location_id": location_id,
        "target_distance_miles": target_distance,
        "impact_prediction": timeline,
    }


@router.get("/risk-score/{location_id}/{fire_id}")
def calculate_risk_score(
    location_id: str,
    fire_id: str,
    distance_miles: float = Query(10.0, ge=0.1, le=100),
    db: Session = Depends(get_db),
):
    """
    Calculate detailed risk score for location given fire.

    Factors:
    - Fire risk (size, containment)
    - Location vulnerability (distance, elevation)
    - Weather conditions
    - Movement trajectory

    Args:
        location_id: Location identifier
        fire_id: Fire identifier
        distance_miles: Distance from location to fire
    """
    location = db.query(Location).filter(Location.location_id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    fire = db.query(Fire).filter(Fire.fire_id == fire_id).first()
    if not fire:
        raise HTTPException(status_code=404, detail="Fire not found")

    scorer = ml_service.risk_scorer

    fire_risk = scorer.calculate_fire_risk_score(fire)
    location_vuln = scorer.calculate_location_vulnerability_score(
        location,
        fire,
        distance_miles,
    )
    spread_rate = ml_service.predictor.predict_spread_rate(
        fire,
        distance_history(db, fire, location),
    )

    try:
        combined_threat = scorer.calculate_combined_threat_score(
            location,
            fire,
            distance_miles,
            spread_rate,
        )
    except TypeError:
        combined_threat = scorer.calculate_combined_threat_score(
            location,
            fire,
            distance_miles,
        )

    threat_level = scorer.get_threat_description(combined_threat)
    recommendations = scorer.get_recommendations(combined_threat)

    return {
        "location_id": location_id,
        "fire_id": fire_id,
        "distance_miles": distance_miles,
        "fire_risk_score": float(fire_risk),
        "location_vulnerability_score": float(location_vuln),
        "combined_threat_score": float(combined_threat),
        "threat_level": threat_level,
        "recommendations": recommendations,
    }


@router.get("/weather-impact/{location_id}")
async def get_weather_impact(
    location_id: str,
    db: Session = Depends(get_db),
):
    """
    Get current and predicted weather impact on fire spread.

    - Current weather
    - Fire weather index
    - Spread multiplier
    - Forecast impact
    """
    location = db.query(Location).filter(Location.location_id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    if location.latitude is None or location.longitude is None:
        raise HTTPException(status_code=400, detail="Location not geocoded")

    weather_service = ml_service.weather_service

    current_weather = await weather_service.get_current_weather(
        location.latitude,
        location.longitude,
    )

    fwi_data = await weather_service.get_fire_weather_indices(
        location.latitude,
        location.longitude,
    )

    forecast = await weather_service.get_forecast(
        location.latitude,
        location.longitude,
        days=3,
    )

    return {
        "location_id": location_id,
        "current_weather": current_weather,
        "fire_weather_index": fwi_data.get("fwi_index") if fwi_data else None,
        "fire_danger_level": fwi_data.get("danger_level") if fwi_data else None,
        "forecast": forecast.get("forecast") if forecast else None,
    }
