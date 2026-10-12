"""Custom application exceptions."""


class FirewatchException(Exception):
    """Base exception for Firewatch application."""

    def __init__(self, message: str, code: str = "FIREWATCH_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class FireDataFetchError(FirewatchException):
    """Raised when fire data fetch fails."""

    def __init__(self, message: str):
        super().__init__(message, code="FIRE_DATA_FETCH_ERROR")


class LocationNotFoundError(FirewatchException):
    """Raised when a monitored location is not found."""

    def __init__(self, location_id: str):
        message = f"Location '{location_id}' not found"
        super().__init__(message, code="LOCATION_NOT_FOUND")


class GeocodeError(FirewatchException):
    """Raised when geocoding fails."""

    def __init__(self, address: str | None = None):
        message = "Could not geocode submitted address"
        super().__init__(message, code="GEOCODE_ERROR")


class WeatherFetchError(FirewatchException):
    """Raised when weather data fetch fails."""

    def __init__(self, message: str):
        super().__init__(message, code="WEATHER_FETCH_ERROR")


class ValidationError(FirewatchException):
    """Raised when data validation fails."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message, code="VALIDATION_ERROR")
        self.field = field


class DatabaseError(FirewatchException):
    """Raised when database operations fail."""

    def __init__(self, message: str):
        super().__init__(message, code="DATABASE_ERROR")
