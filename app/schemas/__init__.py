"""Pydantic schemas for request/response validation."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AlertLevelEnum(StrEnum):
    """Alert level enumeration."""

    SAFE = "SAFE"
    MONITOR = "MONITOR"
    PREPARE = "PREPARE"
    WARNING = "WARNING"
    EVACUATE = "EVACUATE"
    UNKNOWN = "UNKNOWN"


class CardinalDirection(StrEnum):
    """Cardinal directions."""

    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"


# Location Schemas
class LocationBase(BaseModel):
    """Shared location fields."""

    location_id: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    icon: str = Field(default="📍", max_length=10)


class LocationCoordinateFields(BaseModel):
    """Coordinate fields used by create/update/response schemas."""

    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class LocationCreate(LocationBase, LocationCoordinateFields):
    """Schema for creating a location."""

    address: str | None = Field(default=None, min_length=5, max_length=500)
    persist_address: bool = False

    @model_validator(mode="after")
    def validate_location_source(self):
        has_address = bool(self.address)
        has_lat = self.latitude is not None
        has_lon = self.longitude is not None
        if has_lat != has_lon:
            raise ValueError("Latitude and longitude must be provided together")
        if not has_address and not (has_lat and has_lon):
            raise ValueError("Provide either an address or both latitude and longitude")
        return self


class LocationUpdate(LocationCoordinateFields):
    """Schema for updating a location."""

    address: str | None = None
    display_name: str | None = None
    icon: str | None = None
    is_active: bool | None = None
    persist_address: bool = False

    @model_validator(mode="after")
    def validate_partial_coordinates(self):
        has_lat = self.latitude is not None
        has_lon = self.longitude is not None
        if has_lat != has_lon:
            raise ValueError("Latitude and longitude must be provided together")
        return self


class LocationResponse(LocationBase, LocationCoordinateFields):
    """Schema for location response."""

    address: str | None = None
    id: int
    geocoded_at: datetime | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class GeocodePreviewRequest(BaseModel):
    """Request payload for address-to-coordinate preview."""

    address: str = Field(..., min_length=5, max_length=500)


class GeocodePreviewResponse(LocationCoordinateFields):
    """Geocode preview response without persisting the submitted address."""

    latitude: float
    longitude: float
    geocoded_at: datetime


# Fire Schemas
class FireBase(BaseModel):
    """Base fire schema."""

    fire_id: str = Field(..., min_length=1, max_length=200)
    name: str = Field(..., min_length=1, max_length=500)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    acres_burned: float = Field(default=0.0, ge=0)
    percent_contained: float = Field(default=0.0, ge=0, le=100)


class FireCreate(FireBase):
    """Schema for creating a fire."""

    polygon_data: dict[str, Any] | None = None
    extra_metadata: dict[str, Any] | None = None


class FireUpdate(BaseModel):
    """Schema for updating fire data."""

    acres_burned: float | None = None
    percent_contained: float | None = None
    polygon_data: dict[str, Any] | None = None
    extra_metadata: dict[str, Any] | None = None
    is_active: bool | None = None


class FireResponse(FireBase):
    """Schema for fire response."""

    id: int
    last_updated: datetime
    discovered_at: datetime
    polygon_data: dict[str, Any] | None = None
    extra_metadata: dict[str, Any]
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


# Distance History Schemas
class MovementData(BaseModel):
    """Fire movement information."""

    moving_towards: bool = Field(..., description="Is fire moving towards location?")
    speed_mph: float = Field(default=0.0, ge=0, description="Speed in miles per hour")


class DistanceHistoryResponse(BaseModel):
    """Schema for distance history response."""

    fire_id: int
    location_id: int
    distance_miles: float
    bearing_degrees: float | None = None
    direction: str | None = None
    moving_towards: bool
    speed_mph: float
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


# Alert Schemas
class AlertBase(BaseModel):
    """Base alert schema."""

    alert_level: AlertLevelEnum
    distance_miles: float
    reason: str = Field(..., min_length=1, max_length=500)


class AlertCreate(AlertBase):
    """Schema for creating an alert."""

    fire_id: int
    location_id: int


class AlertUpdate(BaseModel):
    """Schema for updating an alert."""

    acknowledged: bool | None = None


class AlertResponse(AlertBase):
    """Schema for alert response."""

    id: int
    fire_id: int
    location_id: int
    timestamp: datetime
    acknowledged: bool
    acknowledged_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


# Threat Assessment Schemas
class ThreatAssessment(BaseModel):
    """Threat assessment for a location."""

    location_id: str
    location_display_name: str
    alert_level: AlertLevelEnum
    nearest_fire_name: str | None = None
    distance_miles: float | None = None
    direction: str | None = None
    acres_burned: float | None = None
    percent_contained: float | None = None
    moving_towards: bool = False
    speed_mph: float = 0.0
    smoke_density: str | None = None
    smoke_distance_miles: float | None = None
    smoke_source: str | None = None
    smoke_observed_at: datetime | None = None
    air_quality_index: int | None = None
    pm25_ug_m3: float | None = None
    air_quality_category: str | None = None
    air_quality_source: str | None = None
    air_quality_observed_at: datetime | None = None
    recommendations: list[str] = []


class FireThreatData(BaseModel):
    """Fire threat data for a location."""

    fire_id: str
    fire_name: str
    latitude: float
    longitude: float
    distance_miles: float
    direction: str
    bearing_degrees: float
    acres_burned: float
    percent_contained: float
    moving_towards: bool
    speed_mph: float
    timestamp: datetime


# Dashboard Schemas
class DashboardData(BaseModel):
    """Complete dashboard data."""

    timestamp: datetime
    threat_assessments: list[ThreatAssessment]
    fires: list[FireResponse]
    recent_alerts: list[AlertResponse]
    all_clear: bool = Field(description="True if all locations are at SAFE level")


class HealthCheck(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    timestamp: datetime
    database: str = "ok"
    services: dict[str, str] = {}


# Weather Schemas
class WeatherData(BaseModel):
    """Weather data for a location."""

    latitude: float
    longitude: float
    temperature: float | None = None
    wind_speed: float | None = None
    wind_direction: float | None = None
    humidity: float | None = None
    conditions: str | None = None
    timestamp: datetime


# Error Response Schemas
class ErrorDetail(BaseModel):
    """Error detail."""

    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    """Error response."""

    error: ErrorDetail
    timestamp: datetime
    request_id: str | None = None
