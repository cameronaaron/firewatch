"""Main FastAPI application with middleware and startup/shutdown handlers."""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.api import alerts_router, fires_router, health_router, locations_router, predictions_router
from app.core.config import get_settings
from app.core.database import init_db
from app.core.exceptions import FirewatchException
from app.core.logging_config import setup_logging
from app.models.database import Alert, DistanceHistory, Location
from app.services import FireDataService, GeocodingService

# Setup logging
settings = get_settings()
setup_logging(level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)
logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Background task for fetching fire data
_fire_fetch_task: asyncio.Task | None = None


async def fetch_fires_periodically(db: Session):
    """Background task to fetch fire data periodically."""
    while True:
        try:
            logger.info("Fetching fire data...")

            # Get active locations
            locations = db.query(Location).filter(Location.is_active).all()
            if not locations:
                logger.warning("No active locations to monitor")
            else:
                # Fetch and process fires
                fires_data = await FireDataService.process_fire_data(db, locations)
                logger.info(f"Fetched {len(fires_data)} nearby fires")

                # Store or update fires in database
                for fire_data in fires_data:
                    db_fire = FireDataService.upsert_fire(db, fire_data)

                    # Record distance history for each location
                    for location_id, distance in fire_data["distances"].items():
                        location = (
                            db.query(Location).filter(Location.location_id == location_id).first()
                        )
                        if location:
                            bearing = fire_data["bearings"].get(location_id, 0)
                            direction = fire_data["directions"].get(location_id, "")
                            movement = fire_data["movement_data"].get(location_id, {})

                            history_entry = DistanceHistory(
                                fire_id=db_fire.id,
                                location_id=location.id,
                                distance_miles=distance,
                                bearing_degrees=bearing,
                                direction=direction,
                                moving_towards=movement.get("moving_towards", False),
                                speed_mph=movement.get("speed", 0.0),
                                observed_acres_burned=fire_data.get("acres_burned"),
                                observed_percent_contained=fire_data.get("percent_contained"),
                                observed_growth_acres_24h=(
                                    fire_data.get("extra_metadata") or {}
                                ).get("growth_acres_24h"),
                            )
                            db.add(history_entry)

                            # Check if alert level changed
                            alert_level = FireDataService.get_alert_level(
                                distance, fire_data["acres_burned"], fire_data["percent_contained"]
                            )

                            # Create alert if not SAFE
                            if alert_level != "SAFE":
                                alert = Alert(
                                    fire_id=db_fire.id,
                                    location_id=location.id,
                                    alert_level=alert_level,
                                    distance_miles=distance,
                                    reason=f"{fire_data['name']} fire is {distance}mi away",
                                )
                                db.add(alert)

                retired = FireDataService.retire_stale_fires(db, datetime.now(UTC))
                db.commit()
                logger.info(
                    "Updated %s fires; retired %s unseen for %sh",
                    len(fires_data),
                    retired,
                    settings.FIRE_STALE_HOURS,
                )

            # Wait before next fetch
            await asyncio.sleep(settings.FIRE_DATA_FETCH_INTERVAL)

        except Exception as e:
            logger.error(f"Error in fire data fetch task: {e}")
            # Wait before retrying
            await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Initialize locations from config if not present
    from app.core.database import SessionLocal

    db = SessionLocal()
    try:
        if settings.MONITORED_LOCATIONS:
            for location_id, location_config in settings.MONITORED_LOCATIONS.items():
                existing = db.query(Location).filter(Location.location_id == location_id).first()
                if not existing:
                    location = Location(
                        location_id=location_id,
                        address=location_config.get("address") or "[stored client-side]",
                        display_name=location_config.get("display_name"),
                        icon=location_config.get("icon", "📍"),
                    )
                    db.add(location)
            db.commit()
            logger.info("Initialized configured monitored locations")
        else:
            logger.info("No default monitored locations configured")
    except Exception as e:
        logger.warning(f"Error initializing locations: {e}")
    finally:
        db.close()

    # Geocode locations
    db = SessionLocal()
    try:
        await GeocodingService.geocode_locations(db)
    except Exception as e:
        logger.warning(f"Error geocoding locations: {e}")
    finally:
        db.close()

    # Start background task
    db = SessionLocal()
    global _fire_fetch_task
    _fire_fetch_task = asyncio.create_task(fetch_fires_periodically(db))
    logger.info("Started fire data fetch task")

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}")
    if _fire_fetch_task:
        _fire_fetch_task.cancel()
        try:
            await _fire_fetch_task
        except asyncio.CancelledError:
            logger.info("Fire data fetch task cancelled")
    db.close()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Real-time wildfire monitoring and evacuation alerts",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Middleware: Trusted hosts
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"] if settings.DEBUG else settings.ALLOWED_HOSTS
)

# Middleware: CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Custom exception handlers
@app.exception_handler(FirewatchException)
async def firewatch_exception_handler(request: Request, exc: FirewatchException):
    """Handle custom Firewatch exceptions."""
    return JSONResponse(
        status_code=400,
        content={
            "error": {"code": exc.code, "message": exc.message},
            "timestamp": datetime.now(UTC).isoformat(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors(),
            },
            "timestamp": datetime.now(UTC).isoformat(),
            "path": str(request.url.path),
        },
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests."""
    start_time = datetime.now(UTC)
    response = await call_next(request)
    duration = (datetime.now(UTC) - start_time).total_seconds()

    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({duration:.2f}s)")
    return response


# Register routers
app.include_router(health_router, prefix=settings.API_V1_PREFIX)
app.include_router(locations_router, prefix=settings.API_V1_PREFIX)
app.include_router(fires_router, prefix=settings.API_V1_PREFIX)
app.include_router(alerts_router, prefix=settings.API_V1_PREFIX)
app.include_router(predictions_router, prefix=settings.API_V1_PREFIX)


# Root endpoint
@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Real-time wildfire monitoring and evacuation alerts",
        "ui": "/app",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": f"{settings.API_V1_PREFIX}/health",
    }


@app.get("/app", include_in_schema=False)
def app_shell() -> FileResponse:
    """Serve the static browser client."""
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
