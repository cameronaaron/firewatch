"""API routes for locations management."""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.database import Location
from app.schemas import (
    GeocodePreviewRequest,
    GeocodePreviewResponse,
    LocationCreate,
    LocationResponse,
    LocationUpdate,
)
from app.services import GeocodingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/locations", tags=["locations"])

CLIENT_SIDE_ADDRESS_PLACEHOLDER = "[stored client-side]"


def _address_for_storage(address: str | None, persist_address: bool) -> str:
    """Keep raw addresses out of server-side storage unless explicitly requested."""
    if persist_address and address:
        return address
    return CLIENT_SIDE_ADDRESS_PLACEHOLDER


def _serialize_location(location: Location) -> LocationResponse:
    """Convert ORM location rows into API-safe responses."""
    address = None if location.address == CLIENT_SIDE_ADDRESS_PLACEHOLDER else location.address
    return LocationResponse(
        id=location.id,
        location_id=location.location_id,
        address=address,
        display_name=location.display_name,
        icon=location.icon,
        latitude=location.latitude,
        longitude=location.longitude,
        geocoded_at=location.geocoded_at,
        is_active=location.is_active,
        created_at=location.created_at,
        updated_at=location.updated_at,
    )


async def _geocode_address_or_422(address: str) -> tuple[float, float]:
    """Resolve an address into coordinates or raise a client-safe HTTP error."""
    try:
        return await GeocodingService.geocode_location(address)
    except Exception as exc:
        logger.warning("Failed to geocode submitted location: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not find that place. Try a street address, city and state, landmark, or "
                "5-digit ZIP code."
            ),
        ) from exc


def _resolved_coordinates(
    latitude: float | None,
    longitude: float | None,
) -> tuple[float | None, float | None, datetime | None]:
    """Return supplied coordinates with a consistent timestamp."""
    if latitude is None or longitude is None:
        return None, None, None
    return latitude, longitude, datetime.now(UTC)


@router.get("", response_model=list[LocationResponse])
def list_locations(active_only: bool = Query(True), db: Session = Depends(get_db)):
    """Get all monitored locations."""
    query = db.query(Location)
    if active_only:
        query = query.filter(Location.is_active)
    locations = query.all()
    return [_serialize_location(location) for location in locations]


@router.get("/{location_id}", response_model=LocationResponse)
def get_location(location_id: str, db: Session = Depends(get_db)):
    """Get specific location by ID."""
    location = db.query(Location).filter(Location.location_id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")
    return _serialize_location(location)


@router.post("/resolve", response_model=GeocodePreviewResponse)
async def resolve_location_address(request: GeocodePreviewRequest):
    """Geocode an address preview without persisting the raw address server-side."""
    lat, lon = await _geocode_address_or_422(request.address)
    return GeocodePreviewResponse(
        latitude=lat,
        longitude=lon,
        geocoded_at=datetime.now(UTC),
    )


@router.post("", response_model=LocationResponse)
async def create_location(location_create: LocationCreate, db: Session = Depends(get_db)):
    """Create a new monitored location."""
    # Check if location already exists
    existing = (
        db.query(Location).filter(Location.location_id == location_create.location_id).first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="Location with this ID already exists")

    lat, lon, geocoded_at = _resolved_coordinates(
        location_create.latitude,
        location_create.longitude,
    )
    if lat is None and lon is None and location_create.address is not None:
        lat, lon = await _geocode_address_or_422(location_create.address)
        geocoded_at = datetime.now(UTC)

    # Create location record
    db_location = Location(
        location_id=location_create.location_id,
        address=_address_for_storage(location_create.address, location_create.persist_address),
        display_name=location_create.display_name,
        icon=location_create.icon,
        latitude=lat,
        longitude=lon,
        geocoded_at=geocoded_at,
    )
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return _serialize_location(db_location)


@router.put("/{location_id}", response_model=LocationResponse)
async def update_location(
    location_id: str, location_update: LocationUpdate, db: Session = Depends(get_db)
):
    """Update a monitored location."""
    location = db.query(Location).filter(Location.location_id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    # Update fields if provided
    if location_update.address is not None:
        lat, lon = await _geocode_address_or_422(location_update.address)
        location.latitude = lat
        location.longitude = lon
        location.geocoded_at = datetime.now(UTC)
        location.address = _address_for_storage(
            location_update.address, location_update.persist_address
        )

    if location_update.latitude is not None and location_update.longitude is not None:
        location.latitude = location_update.latitude
        location.longitude = location_update.longitude
        location.geocoded_at = datetime.now(UTC)
        if location_update.address is None:
            location.address = _address_for_storage(None, location_update.persist_address)

    if location_update.display_name is not None:
        location.display_name = location_update.display_name
    if location_update.icon is not None:
        location.icon = location_update.icon
    if location_update.is_active is not None:
        location.is_active = location_update.is_active

    db.commit()
    db.refresh(location)
    return _serialize_location(location)


@router.delete("/{location_id}")
def delete_location(location_id: str, db: Session = Depends(get_db)):
    """Delete a monitored location."""
    location = db.query(Location).filter(Location.location_id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    db.delete(location)
    db.commit()
    return {"message": "Location deleted successfully"}
