"""
Authentication module for Garmin Connect integration.
Handles secure token storage and Garmin Connect API authentication.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import structlog
from garminconnect import Garmin
import garminconnect
from pydantic import BaseModel

from .config import GarminMCPConfig
from .exceptions import AuthenticationError, GarminConnectionError

logger = structlog.get_logger(__name__)


class AuthStatus(BaseModel):
    """Authentication status information."""
    authenticated: bool
    last_login: Optional[datetime] = None
    token_expires: Optional[datetime] = None
    username: Optional[str] = None
    error_message: Optional[str] = None
    connection_quality: str = "unknown"  # good, degraded, poor, failed


class GarminAuthenticator:
    """Handles Garmin Connect authentication with secure token management."""
    
    def __init__(self, config: GarminMCPConfig):
        self.config = config
        self.client: Optional[Garmin] = None
        self.token_file = config.token_file_path
        self._last_auth_check: Optional[datetime] = None
        self._auth_status = AuthStatus(authenticated=False)
        
        # Ensure token directory exists
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the authenticator and attempt login."""
        logger.info("Initializing Garmin authenticator")
        
        try:
            # Check garminconnect library version
            version = getattr(garminconnect, '__version__', 'unknown')
            logger.debug("Using garminconnect library", version=version)
            
            # Create Garmin client
            email, password = self.config.get_garmin_credentials()
            self.client = Garmin(email, password)
            
            # Attempt authentication
            await self._authenticate()
            
            logger.info("Garmin authenticator initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Garmin authenticator", error=str(e))
            self._auth_status = AuthStatus(
                authenticated=False,
                error_message=str(e)
            )
            raise AuthenticationError(f"Authentication initialization failed: {e}")
    
    async def _authenticate(self) -> bool:
        """Perform authentication with Garmin Connect."""
        if not self.client:
            raise RuntimeError("Garmin client not initialized")
        
        logger.info("Authenticating with Garmin Connect")
        
        # Step 1: Try to load saved tokens
        if await self._try_token_authentication():
            logger.info("Token authentication successful")
            return True
        
        # Step 2: Perform fresh credential authentication
        return await self._perform_credential_authentication()
    
    async def _try_token_authentication(self) -> bool:
        """Try to authenticate using saved tokens."""
        if not self.token_file.exists():
            logger.debug("No token file found, skipping token authentication")
            return False
        
        try:
            logger.info("Attempting token-based authentication")
            
            # Load tokens from file
            tokens_data = await self._load_tokens()
            if not tokens_data:
                logger.warning("Token file exists but is empty or invalid")
                return False
            
            # Load tokens into Garmin client
            self.client.garth.loads(tokens_data)

            # garminconnect.login() sets these after garth.loads; loading
            # tokens directly skips that, leaving display_name None and
            # breaking endpoints that embed it in the URL (e.g. heart rate).
            profile = self.client.garth.profile or {}
            self.client.display_name = profile.get("displayName")
            self.client.full_name = profile.get("fullName")

            # Test the tokens with a simple API call
            await self._test_connection()
            
            # Update auth status
            self._auth_status = AuthStatus(
                authenticated=True,
                last_login=datetime.now(),
                username=self.config.garmin_email,
                connection_quality="good"
            )
            
            logger.info("Token authentication successful")
            return True
            
        except Exception as e:
            logger.warning("Token authentication failed", error=str(e))
            
            # Clean up invalid tokens
            await self._cleanup_invalid_tokens()
            return False
    
    async def _perform_credential_authentication(self) -> bool:
        """Perform fresh credential-based authentication."""
        if not self.client:
            raise RuntimeError("Garmin client not initialized")
        
        logger.info("Performing credential-based authentication")
        
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Authentication attempt {attempt}/{max_attempts}")
                
                # Perform login
                self.client.login()
                
                # Test the connection
                await self._test_connection()
                
                # Save tokens for future use
                await self._save_tokens()
                
                # Update auth status
                self._auth_status = AuthStatus(
                    authenticated=True,
                    last_login=datetime.now(),
                    username=self.config.garmin_email,
                    connection_quality="good"
                )
                
                logger.info("Credential authentication successful")
                return True
                
            except Exception as e:
                logger.error(f"Authentication attempt {attempt} failed", error=str(e))
                
                if attempt == max_attempts:
                    error_msg = (
                        f"Authentication failed after {max_attempts} attempts. "
                        f"Please verify your Garmin Connect credentials. "
                        f"Last error: {e}"
                    )
                    self._auth_status = AuthStatus(
                        authenticated=False,
                        error_message=error_msg,
                        connection_quality="failed"
                    )
                    raise AuthenticationError(error_msg)
                else:
                    # Brief delay between attempts
                    await asyncio.sleep(2)
        
        return False
    
    async def _test_connection(self) -> None:
        """Test the Garmin Connect connection with a simple API call."""
        if not self.client:
            raise RuntimeError("Garmin client not initialized")
        
        try:
            # Try a simple API call to verify authentication
            profile = self.client.get_full_name()
            logger.debug("Connection test successful", profile=profile)
            
        except Exception as e:
            logger.error("Connection test failed", error=str(e))
            raise GarminConnectionError(f"Connection test failed: {e}")
    
    async def _save_tokens(self) -> None:
        """Save authentication tokens to file."""
        if not self.client:
            return
        
        try:
            # Get token string from garth
            tokens_string = self.client.garth.dumps()
            if not tokens_string:
                logger.warning("No tokens to save")
                return
            
            # Write to file atomically
            temp_file = self.token_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                f.write(tokens_string)
            
            # Atomic rename
            temp_file.rename(self.token_file)
            
            # Set restrictive permissions
            os.chmod(self.token_file, 0o600)
            
            logger.info("Authentication tokens saved", 
                       file=str(self.token_file),
                       size=len(tokens_string))
            
        except Exception as e:
            logger.error("Failed to save tokens", error=str(e))
            # Don't raise - this is not critical
    
    async def _load_tokens(self) -> Optional[str]:
        """Load authentication tokens from file."""
        try:
            if not self.token_file.exists():
                return None
            
            with open(self.token_file, 'r') as f:
                tokens_string = f.read().strip()
            
            if not tokens_string:
                logger.warning("Token file is empty")
                return None
            
            logger.debug("Tokens loaded from file", 
                        file=str(self.token_file),
                        size=len(tokens_string))
            
            return tokens_string
            
        except Exception as e:
            logger.error("Failed to load tokens", error=str(e))
            return None
    
    async def _cleanup_invalid_tokens(self) -> None:
        """Clean up invalid token files."""
        try:
            if self.token_file.exists():
                self.token_file.unlink()
                logger.info("Removed invalid token file", file=str(self.token_file))
        except Exception as e:
            logger.error("Failed to cleanup invalid tokens", error=str(e))
    
    async def get_auth_status(self) -> AuthStatus:
        """Get current authentication status."""
        # Refresh status if it's been a while
        if (self._last_auth_check is None or 
            datetime.now() - self._last_auth_check > timedelta(minutes=5)):
            await self._refresh_auth_status()
        
        return self._auth_status
    
    async def _refresh_auth_status(self) -> None:
        """Refresh authentication status by testing connection."""
        self._last_auth_check = datetime.now()
        
        if not self.client or not self._auth_status.authenticated:
            return
        
        try:
            # Quick connection test
            await self._test_connection()
            
            # Update connection quality
            self._auth_status.connection_quality = "good"
            
        except Exception as e:
            logger.warning("Auth status refresh failed", error=str(e))
            
            # Try to re-authenticate
            try:
                await self._authenticate()
            except Exception as re_auth_error:
                logger.error("Re-authentication failed", error=str(re_auth_error))
                self._auth_status = AuthStatus(
                    authenticated=False,
                    error_message=str(re_auth_error),
                    connection_quality="failed"
                )
    
    async def ensure_authenticated(self) -> None:
        """Ensure we have valid authentication, re-authenticate if needed."""
        status = await self.get_auth_status()
        
        if not status.authenticated:
            logger.info("Authentication required, attempting login")
            await self._authenticate()
            
            # Verify authentication succeeded
            if not self._auth_status.authenticated:
                raise AuthenticationError(
                    f"Authentication failed: {self._auth_status.error_message}"
                )
    
    def get_client(self) -> Garmin:
        """Get the authenticated Garmin client."""
        if not self.client:
            raise RuntimeError("Garmin client not initialized")
        
        if not self._auth_status.authenticated:
            raise AuthenticationError("Not authenticated with Garmin Connect")
        
        return self.client
    
    async def logout(self) -> None:
        """Logout and cleanup tokens."""
        logger.info("Logging out from Garmin Connect")
        
        try:
            # Cleanup tokens
            await self._cleanup_invalid_tokens()
            
            # Reset auth status
            self._auth_status = AuthStatus(authenticated=False)
            
            # Reset client
            self.client = None
            
            logger.info("Logout completed")
            
        except Exception as e:
            logger.error("Error during logout", error=str(e))
    
    async def get_profile_info(self) -> Dict[str, Any]:
        """Get basic profile information for authentication verification."""
        await self.ensure_authenticated()
        client = self.get_client()
        
        try:
            profile_data = {
                "full_name": client.get_full_name(),
                "unit_system": client.get_unit_system(),
            }
            
            # Try to get additional profile info if available
            try:
                profile_data["display_name"] = client.get_user_summary()["displayName"]
            except:
                pass
            
            return profile_data
            
        except Exception as e:
            logger.error("Failed to get profile info", error=str(e))
            raise GarminConnectionError(f"Failed to get profile: {e}")