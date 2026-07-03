"""
Configuration management for Garmin MCP Server.
"""

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class GarminMCPConfig(BaseSettings):
    """Configuration settings for Garmin MCP Server."""
    
    # Garmin Connect credentials
    garmin_email: str = Field(..., description="Garmin Connect email/username")
    garmin_password: str = Field(..., description="Garmin Connect password")
    
    # MCP Server configuration
    mcp_transport: Literal["stdio", "http"] = Field(
        default="stdio", 
        description="MCP transport method"
    )
    mcp_host: str = Field(default="localhost", description="Host for HTTP transport")
    mcp_port: int = Field(default=8000, description="Port for HTTP transport")
    
    # Logging configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", 
        description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="text",
        description="Log format"
    )
    
    # Security configuration (for HTTP transport)
    oauth_client_id: Optional[str] = Field(
        default=None, 
        description="OAuth client ID for HTTP transport"
    )
    oauth_client_secret: Optional[str] = Field(
        default=None, 
        description="OAuth client secret for HTTP transport"
    )
    jwt_secret_key: Optional[str] = Field(
        default=None,
        description="JWT secret key for HTTP authentication"
    )
    
    # Cache configuration
    cache_ttl: int = Field(
        default=300, 
        description="Cache TTL in seconds",
        ge=0
    )
    cache_type: Literal["memory", "redis"] = Field(
        default="memory",
        description="Cache backend type"
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis URL for Redis cache"
    )
    
    # Rate limiting
    rate_limit_requests: int = Field(
        default=60,
        description="Requests per minute per client",
        ge=1
    )
    rate_limit_window: int = Field(
        default=60,
        description="Rate limit window in seconds",
        ge=1
    )
    
    # Performance settings
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum concurrent requests",
        ge=1,
        le=100
    )
    request_timeout: int = Field(
        default=30,
        description="Request timeout in seconds",
        ge=1,
        le=300
    )
    garmin_api_rate_limit: int = Field(
        default=10,
        description="Garmin API requests per minute",
        ge=1,
        le=60
    )
    
    # Data settings
    default_date_range: int = Field(
        default=7,
        description="Default number of days for data queries",
        ge=1,
        le=90
    )
    max_date_range: int = Field(
        default=90,
        description="Maximum allowed date range",
        ge=1,
        le=365
    )
    
    # Monitoring (optional)
    enable_metrics: bool = Field(
        default=False,
        description="Enable Prometheus metrics"
    )
    metrics_port: int = Field(
        default=9000,
        description="Port for metrics endpoint"
    )
    
    # Development settings
    development_mode: bool = Field(
        default=False,
        description="Enable development features"
    )
    debug_garmin_api: bool = Field(
        default=False,
        description="Enable detailed Garmin API logging"
    )
    
    # Token storage
    token_file_path: Path = Field(
        default=Path(".garmin_tokens"),
        description="Path to Garmin token storage file"
    )
    
    @field_validator("max_date_range")
    @classmethod
    def validate_max_date_range(cls, v, info):
        """Ensure max_date_range is greater than or equal to default_date_range."""
        if hasattr(info.data, 'get'):
            default_range = info.data.get("default_date_range", 7)
            if v < default_range:
                raise ValueError("max_date_range must be >= default_date_range")
        return v
    
    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret_for_http(cls, v, info):
        """Require JWT secret for HTTP transport."""
        if hasattr(info.data, 'get'):
            transport = info.data.get("mcp_transport")
            if transport == "http" and not v:
                raise ValueError("jwt_secret_key required for HTTP transport")
        return v
    
    @field_validator("oauth_client_id", "oauth_client_secret")
    @classmethod
    def validate_oauth_for_http(cls, v, info):
        """Require OAuth credentials for HTTP transport."""
        if hasattr(info.data, 'get'):
            transport = info.data.get("mcp_transport")
            if transport == "http" and not v:
                field_name = info.field_name
                raise ValueError(f"{field_name} required for HTTP transport")
        return v
    
    class Config:
        env_prefix = ""
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
        # The .env may be shared with the host app (OLLAMA_HOST etc.)
        extra = "ignore"
        
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "GarminMCPConfig":
        """Create configuration from environment variables and .env file."""
        if env_file:
            return cls(_env_file=env_file)
        return cls()
    
    def get_garmin_credentials(self) -> tuple[str, str]:
        """Get Garmin Connect credentials as tuple."""
        return (self.garmin_email, self.garmin_password)
    
    def is_http_transport(self) -> bool:
        """Check if using HTTP transport."""
        return self.mcp_transport == "http"
    
    def is_stdio_transport(self) -> bool:
        """Check if using STDIO transport."""
        return self.mcp_transport == "stdio"
    
    def get_log_config(self) -> dict:
        """Get logging configuration dictionary."""
        return {
            "level": self.log_level,
            "format": self.log_format,
            "development": self.development_mode,
        }
    
    def get_cache_config(self) -> dict:
        """Get cache configuration dictionary."""
        config = {
            "type": self.cache_type,
            "ttl": self.cache_ttl,
        }
        
        if self.cache_type == "redis":
            config["redis_url"] = self.redis_url
            
        return config
    
    def get_rate_limit_config(self) -> dict:
        """Get rate limiting configuration dictionary."""
        return {
            "requests": self.rate_limit_requests,
            "window": self.rate_limit_window,
            "garmin_api_limit": self.garmin_api_rate_limit,
        }
    
    def validate_required_settings(self) -> None:
        """Validate that all required settings are present."""
        if not self.garmin_email or not self.garmin_password:
            raise ValueError(
                "Garmin credentials required: GARMIN_EMAIL and GARMIN_PASSWORD"
            )
        
        if self.is_http_transport():
            if not self.oauth_client_id or not self.oauth_client_secret:
                raise ValueError(
                    "OAuth credentials required for HTTP transport: "
                    "OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET"
                )
            if not self.jwt_secret_key:
                raise ValueError(
                    "JWT secret key required for HTTP transport: JWT_SECRET_KEY"
                )


class GarminAPIConfig(BaseModel):
    """Configuration specific to Garmin API interactions."""
    
    base_url: str = Field(default="https://connect.garmin.com")
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; GarminMCPServer/1.0)"
    )
    timeout: int = Field(default=30, ge=1, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # API endpoints
    auth_url: str = Field(default="https://sso.garmin.com/sso/signin")
    api_url: str = Field(default="https://connect.garmin.com/modern/proxy")
    
    @classmethod
    def create_default(cls) -> "GarminAPIConfig":
        """Create default Garmin API configuration."""
        return cls()


# Global configuration instance (will be initialized on server start)
_config: Optional[GarminMCPConfig] = None

def get_config() -> GarminMCPConfig:
    """Get the global configuration instance."""
    if _config is None:
        raise RuntimeError("Configuration not initialized")
    return _config

def set_config(config: GarminMCPConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config