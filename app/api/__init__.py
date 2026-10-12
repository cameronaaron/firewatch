"""API routes package."""

from .alerts import router as alerts_router
from .fires import router as fires_router
from .health import router as health_router
from .locations import router as locations_router
from .predictions import router as predictions_router

__all__ = [
    "locations_router",
    "fires_router",
    "alerts_router",
    "health_router",
    "predictions_router",
]
