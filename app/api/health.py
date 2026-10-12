"""Health check and status routes."""

from datetime import UTC, datetime

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas import HealthCheck

router = APIRouter(tags=["health"])

settings = get_settings()


@router.get("/health", response_model=HealthCheck)
def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.now(UTC),
        database="ok",
        services={
            "fire_data": "ok",
            "geocoding": "ok",
        },
    )


@router.get("/status")
def status():
    """Get application status."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(UTC).isoformat(),
    }
