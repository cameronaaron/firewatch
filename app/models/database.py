"""SQLAlchemy ORM models for persistent data storage."""

from datetime import UTC, datetime
from types import SimpleNamespace

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Columns with a default are typed non-null: the app always fills them.
    Their DDL keeps nullable=True so existing databases need no migration."""


def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


class Location(Base):
    """Monitored location model."""

    __tablename__ = "locations"
    __table_args__ = (
        Index("idx_location_id", "location_id"),
        CheckConstraint("length(trim(location_id)) > 0", name="ck_location_id_nonempty"),
        CheckConstraint("length(trim(display_name)) > 0", name="ck_location_display_name_nonempty"),
        CheckConstraint("length(trim(address)) > 0", name="ck_location_address_nonempty"),
        CheckConstraint(
            "((latitude IS NULL AND longitude IS NULL) OR "
            "(latitude IS NOT NULL AND longitude IS NOT NULL AND "
            "latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180))",
            name="ck_location_lat_lon_pair_and_range",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    location_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    address: Mapped[str] = mapped_column(String(500), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    icon: Mapped[str] = mapped_column(String(10), default="📍", nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    geocoded_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now, nullable=True
    )

    history: Mapped[list["DistanceHistory"]] = relationship(
        "DistanceHistory", back_populates="location", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["Alert"]] = relationship(
        "Alert", back_populates="location", cascade="all, delete-orphan"
    )

    @property
    def coordinates(self) -> tuple[float, float] | None:
        """(latitude, longitude), or None until the address is geocoded."""
        if self.latitude is None or self.longitude is None:
            return None
        return self.latitude, self.longitude

    def __repr__(self) -> str:
        return f"<Location {self.location_id} ({self.display_name})>"


class Fire(Base):
    """Active fire record."""

    __tablename__ = "fires"
    __table_args__ = (
        Index("idx_fire_id", "fire_id"),
        Index("idx_latitude_longitude", "latitude", "longitude"),
        CheckConstraint("length(trim(fire_id)) > 0", name="ck_fire_id_nonempty"),
        CheckConstraint("length(trim(name)) > 0", name="ck_fire_name_nonempty"),
        CheckConstraint("latitude BETWEEN -90 AND 90", name="ck_fire_latitude_range"),
        CheckConstraint("longitude BETWEEN -180 AND 180", name="ck_fire_longitude_range"),
        CheckConstraint("acres_burned >= 0", name="ck_fire_acres_nonnegative"),
        CheckConstraint(
            "percent_contained >= 0 AND percent_contained <= 100",
            name="ck_fire_percent_contained_range",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    fire_id: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    acres_burned: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    percent_contained: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now, nullable=True
    )
    discovered_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=True)
    polygon_data: Mapped[dict | None] = mapped_column(JSON, default=None)
    extra_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=True)
    bearing_degrees = None

    distances: Mapped[list["DistanceHistory"]] = relationship(
        "DistanceHistory", back_populates="fire", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["Alert"]] = relationship(
        "Alert", back_populates="fire", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Fire {self.name} ({self.fire_id})>"


class DistanceHistory(Base):
    """Historical distance tracking for fire-location pairs."""

    __tablename__ = "distance_history"
    __table_args__ = (
        Index("idx_fire_location_timestamp", "fire_id", "location_id", "timestamp"),
        Index("idx_timestamp", "timestamp"),
        CheckConstraint("distance_miles >= 0", name="ck_distance_nonnegative"),
        CheckConstraint("speed_mph >= 0", name="ck_speed_nonnegative"),
        CheckConstraint(
            "bearing_degrees IS NULL OR (bearing_degrees >= 0 AND bearing_degrees < 360)",
            name="ck_bearing_range",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    fire_id: Mapped[int] = mapped_column(Integer, ForeignKey("fires.id"), nullable=False)
    location_id: Mapped[int] = mapped_column(Integer, ForeignKey("locations.id"), nullable=False)
    distance_miles: Mapped[float] = mapped_column(Float, nullable=False)
    bearing_degrees: Mapped[float | None] = mapped_column(Float, nullable=True)
    direction: Mapped[str | None] = mapped_column(String(50), nullable=True)  # e.g., "NE"
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, index=True, nullable=True
    )
    moving_towards: Mapped[bool] = mapped_column(Boolean, default=False, nullable=True)
    speed_mph: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    observed_acres_burned: Mapped[float | None] = mapped_column(Float, nullable=True)
    observed_percent_contained: Mapped[float | None] = mapped_column(Float, nullable=True)
    observed_growth_acres_24h: Mapped[float | None] = mapped_column(Float, nullable=True)

    fire: Mapped["Fire"] = relationship("Fire", back_populates="distances")
    location: Mapped["Location"] = relationship("Location", back_populates="history")

    def __repr__(self) -> str:
        return (
            f"<DistanceHistory fire_id={self.fire_id} location_id={self.location_id} "
            f"dist={self.distance_miles}mi>"
        )


class Alert(Base):
    """Alert events for monitored locations."""

    __tablename__ = "alerts"
    __table_args__ = (
        Index("idx_alert_location_timestamp", "location_id", "timestamp"),
        Index("idx_alert_fire_timestamp", "fire_id", "timestamp"),
        Index("idx_alert_level_timestamp", "alert_level", "timestamp"),
        CheckConstraint("distance_miles >= 0", name="ck_alert_distance_nonnegative"),
        CheckConstraint(
            "alert_level IN ('SAFE','MONITOR','PREPARE','WARNING','EVACUATE')",
            name="ck_alert_level_allowed",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    fire_id: Mapped[int] = mapped_column(Integer, ForeignKey("fires.id"), nullable=False)
    location_id: Mapped[int] = mapped_column(Integer, ForeignKey("locations.id"), nullable=False)
    # SAFE, MONITOR, PREPARE, WARNING or EVACUATE (see ck_alert_level_allowed).
    alert_level: Mapped[str] = mapped_column(String(20), nullable=False)
    distance_miles: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(String(500), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, index=True, nullable=True
    )
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    fire: Mapped["Fire"] = relationship("Fire", back_populates="alerts")
    location: Mapped["Location"] = relationship("Location", back_populates="alerts")

    def __repr__(self) -> str:
        return f"<Alert {self.alert_level} location_id={self.location_id}>"


class WeatherCache(Base):
    """Cached weather data to reduce API calls."""

    __tablename__ = "weather_cache"
    __table_args__ = (
        Index("idx_lat_lon_timestamp", "latitude", "longitude", "timestamp"),
        CheckConstraint("latitude BETWEEN -90 AND 90", name="ck_weather_latitude_range"),
        CheckConstraint("longitude BETWEEN -180 AND 180", name="ck_weather_longitude_range"),
        CheckConstraint(
            "humidity IS NULL OR (humidity >= 0 AND humidity <= 100)",
            name="ck_weather_humidity_range",
        ),
        CheckConstraint(
            "wind_speed IS NULL OR wind_speed >= 0", name="ck_weather_wind_nonnegative"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_direction: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity: Mapped[float | None] = mapped_column(Float, nullable=True)
    conditions: Mapped[str | None] = mapped_column(String(200), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, index=True, nullable=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    def __repr__(self) -> str:
        return f"<WeatherCache {self.latitude},{self.longitude} {self.temperature}°F>"


# A Fire row, or a detached copy carrying the same attributes (see
# MLFireAnalysisService._build_effective_fire), which must not touch the row.
FireLike = Fire | SimpleNamespace
