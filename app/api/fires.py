"""API routes for fire data."""

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.database import Alert, Fire
from app.schemas import AlertResponse, FireResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fires", tags=["fires"])


@router.get("", response_model=list[FireResponse])
def list_fires(
    active_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Get list of fires."""
    query = db.query(Fire)
    if active_only:
        query = query.filter(Fire.is_active)
    fires = query.order_by(Fire.last_updated.desc()).limit(limit).all()
    return fires


@router.get("/recent", response_model=list[AlertResponse])
def get_recent_alerts(
    hours: int = Query(24, ge=1, le=720),  # Max 30 days
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Get recent alerts across all locations."""
    since = datetime.now(UTC) - timedelta(hours=hours)
    alerts = (
        db.query(Alert)
        .filter(Alert.timestamp >= since)
        .order_by(Alert.timestamp.desc())
        .limit(limit)
        .all()
    )

    return alerts


@router.get("/{fire_id}", response_model=FireResponse)
def get_fire(fire_id: str, db: Session = Depends(get_db)):
    """Get specific fire by ID."""
    fire = db.query(Fire).filter(Fire.fire_id == fire_id).first()
    if not fire:
        raise HTTPException(status_code=404, detail="Fire not found")
    return fire


@router.get("/{fire_id}/alerts", response_model=list[AlertResponse])
def get_fire_alerts(fire_id: str, hours: int = Query(24, ge=1), db: Session = Depends(get_db)):
    """Get recent alerts for a fire."""
    fire = db.query(Fire).filter(Fire.fire_id == fire_id).first()
    if not fire:
        raise HTTPException(status_code=404, detail="Fire not found")

    since = datetime.now(UTC) - timedelta(hours=hours)
    alerts = (
        db.query(Alert)
        .filter(Alert.fire_id == fire.id, Alert.timestamp >= since)
        .order_by(Alert.timestamp.desc())
        .all()
    )

    return alerts
