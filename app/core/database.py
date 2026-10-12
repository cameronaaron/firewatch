"""Database configuration and session management."""

import logging
from collections.abc import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.models.database import Base

logger = logging.getLogger(__name__)

settings = get_settings()

# Create engine
if "sqlite" in settings.DATABASE_URL:
    # SQLite specific configuration
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.DATABASE_ECHO,
    )
else:  # pragma: no cover
    # PostgreSQL/MySQL configuration
    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,  # Recycle connections every hour
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        if "sqlite" in settings.DATABASE_URL:
            _ensure_sqlite_distance_history_columns()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def _ensure_sqlite_distance_history_columns() -> None:
    """Add newly required historical fire-state columns to existing SQLite databases."""
    # Literal statements: nothing from outside reaches this DDL.
    expected_columns = {
        "observed_acres_burned": (
            "ALTER TABLE distance_history ADD COLUMN observed_acres_burned FLOAT"
        ),
        "observed_percent_contained": (
            "ALTER TABLE distance_history ADD COLUMN observed_percent_contained FLOAT"
        ),
        "observed_growth_acres_24h": (
            "ALTER TABLE distance_history ADD COLUMN observed_growth_acres_24h FLOAT"
        ),
    }

    with engine.begin() as connection:
        existing_columns = {
            row[1] for row in connection.execute(text("PRAGMA table_info(distance_history)"))
        }
        if not existing_columns:
            # No table yet: create_all builds it with every column.
            return
        for column_name, statement in expected_columns.items():
            if column_name not in existing_columns:
                connection.execute(text(statement))


def get_db() -> Generator[Session, None, None]:
    """Get database session for dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Enable foreign keys for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable foreign keys for SQLite."""
    if "sqlite" in settings.DATABASE_URL:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
