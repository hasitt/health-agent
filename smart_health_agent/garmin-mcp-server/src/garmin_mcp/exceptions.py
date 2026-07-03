"""
Custom exceptions for Garmin MCP Server.
"""


class GarminMCPError(Exception):
    """Base exception for Garmin MCP Server errors."""
    pass


class AuthenticationError(GarminMCPError):
    """Raised when authentication with Garmin Connect fails."""
    pass


class GarminConnectionError(GarminMCPError):
    """Raised when connection to Garmin Connect fails."""
    pass


class DataFetchError(GarminMCPError):
    """Raised when data fetching from Garmin Connect fails."""
    pass


class ValidationError(GarminMCPError):
    """Raised when input validation fails."""
    pass


class ConfigurationError(GarminMCPError):
    """Raised when configuration is invalid."""
    pass


class RateLimitError(GarminMCPError):
    """Raised when rate limits are exceeded."""
    pass


class CacheError(GarminMCPError):
    """Raised when cache operations fail."""
    pass