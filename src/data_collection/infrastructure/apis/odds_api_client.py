"""
The Odds API Client

Async HTTP client for The Odds API with:
- Quota tracking (500 requests/month on free tier)
- Alert system for quota management
- Multi-bookmaker odds parsing
- Line movement tracking
- Support for multiple markets
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.shared.config.settings import settings
from src.shared.logging.logger import get_logger
from src.shared.exceptions.custom_exceptions import (
    APIQuotaExceededError,
    APIAuthenticationError,
    APINotFoundError,
    APIServerError,
)
from src.data_collection.application.dtos.api_schemas import (
    OddsResponseSchema,
    EventSchema,
    SportSchema,
)

logger = get_logger(__name__)


class OddsMarket(str, Enum):
    """Available betting markets"""
    H2H = "h2h"  # Head to head (1X2)
    SPREADS = "spreads"  # Handicap
    TOTALS = "totals"  # Over/Under
    BTTS = "btts"  # Both teams to score


class OddsRegion(str, Enum):
    """Bookmaker regions"""
    UK = "uk"
    EU = "eu"
    US = "us"
    AU = "au"


class Sport(str, Enum):
    """Supported sports"""
    SOCCER_EPL = "soccer_epl"  # Premier League
    SOCCER_SPAIN_LA_LIGA = "soccer_spain_la_liga"
    SOCCER_ITALY_SERIE_A = "soccer_italy_serie_a"
    SOCCER_GERMANY_BUNDESLIGA = "soccer_germany_bundesliga"
    SOCCER_UEFA_CHAMPS_LEAGUE = "soccer_uefa_champs_league"


class OddsAPIClient:
    """
    Async client for The Odds API
    
    Features:
    - Quota tracking (500 requests/month free tier)
    - Alert when approaching quota limit (90% threshold)
    - Multi-bookmaker odds comparison
    - Line movement tracking
    - Historical odds support
    
    Usage:
        async with OddsAPIClient() as client:
            odds = await client.get_odds(Sport.SOCCER_EPL)
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    QUOTA_WARNING_THRESHOLD = 0.9  # Alert at 90% usage
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize The Odds API client
        
        Args:
            api_key: API key (defaults to settings.ODDS_API_KEY)
        """
        self.api_key = api_key or settings.ODDS_API_KEY
        
        # HTTP client configuration
        self.client: Optional[httpx.AsyncClient] = None
        
        # Quota tracking
        self.requests_used = 0
        self.requests_remaining = 500  # Free tier default
        self.quota_limit = 500
        
        # Request statistics
        self.request_count = 0
        self.error_count = 0
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=30.0,
        )
        logger.info("The Odds API client initialized")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()
        logger.info(
            f"The Odds API client closed. "
            f"Requests: {self.request_count}, "
            f"Quota remaining: {self.requests_remaining}/{self.quota_limit}"
        )
    
    def _update_quota(self, response: httpx.Response) -> None:
        """
        Update quota information from response headers
        
        Args:
            response: HTTP response with quota headers
        """
        # The Odds API returns quota info in headers
        if "x-requests-used" in response.headers:
            self.requests_used = int(response.headers["x-requests-used"])
        
        if "x-requests-remaining" in response.headers:
            self.requests_remaining = int(response.headers["x-requests-remaining"])
        
        # Calculate quota usage percentage
        quota_usage = self.requests_used / self.quota_limit
        
        # Alert if approaching limit
        if quota_usage >= self.QUOTA_WARNING_THRESHOLD:
            logger.warning(
                f"⚠️ API quota at {quota_usage*100:.1f}% "
                f"({self.requests_used}/{self.quota_limit}). "
                f"Remaining: {self.requests_remaining}"
            )
        
        # Raise error if quota exceeded
        if self.requests_remaining <= 0:
            raise APIQuotaExceededError(
                f"API quota exceeded. Used: {self.requests_used}/{self.quota_limit}"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    )
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and quota tracking
        
        Args:
            endpoint: API endpoint
            params: Query parameters
        
        Returns:
            JSON response as dictionary
        
        Raises:
            APIQuotaExceededError: Quota limit reached
            APIAuthenticationError: Invalid API key
            APINotFoundError: Resource not found
            APIServerError: Server error
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        # Add API key to params
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        
        try:
            logger.debug(f"GET {endpoint} with params {params}")
            
            response = await self.client.get(endpoint, params=params)
            self.request_count += 1
            
            # Update quota tracking
            self._update_quota(response)
            
            # Handle HTTP errors
            if response.status_code == 401:
                self.error_count += 1
                logger.error("Authentication failed")
                raise APIAuthenticationError("Invalid API key")
            
            elif response.status_code == 404:
                self.error_count += 1
                logger.error(f"Resource not found: {endpoint}")
                raise APINotFoundError(f"Resource not found: {endpoint}")
            
            elif response.status_code >= 500:
                self.error_count += 1
                logger.error(f"Server error: {response.status_code}")
                raise APIServerError(f"Server error: {response.status_code}")
            
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Response received: {len(str(data))} bytes")
            
            return data
        
        except httpx.TimeoutException as e:
            self.error_count += 1
            logger.error(f"Request timeout: {endpoint}")
            raise
        
        except httpx.NetworkError as e:
            self.error_count += 1
            logger.error(f"Network error: {e}")
            raise
    
    async def get_sports(self) -> List[SportSchema]:
        """
        Get list of available sports
        
        Returns:
            List of SportSchema objects
        """
        endpoint = "/sports"
        data = await self._make_request(endpoint)
        
        logger.info(f"Fetched {len(data)} available sports")
        
        return [SportSchema(**sport) for sport in data]
    
    async def get_odds(
        self,
        sport: Sport,
        regions: Optional[List[OddsRegion]] = None,
        markets: Optional[List[OddsMarket]] = None,
        odds_format: str = "decimal",
        date_format: str = "iso",
    ) -> OddsResponseSchema:
        """
        Get current odds for a sport
        
        Args:
            sport: Sport identifier
            regions: Bookmaker regions (defaults to EU)
            markets: Betting markets (defaults to h2h)
            odds_format: Odds format (decimal, american)
            date_format: Date format (iso, unix)
        
        Returns:
            OddsResponseSchema with odds data
        
        Example:
            odds = await client.get_odds(
                sport=Sport.SOCCER_EPL,
                regions=[OddsRegion.EU],
                markets=[OddsMarket.H2H, OddsMarket.TOTALS]
            )
        """
        if regions is None:
            regions = [OddsRegion.EU]
        if markets is None:
            markets = [OddsMarket.H2H]
        
        params = {
            "regions": ",".join([r.value for r in regions]),
            "markets": ",".join([m.value for m in markets]),
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        
        endpoint = f"/sports/{sport.value}/odds"
        data = await self._make_request(endpoint, params)
        
        logger.info(
            f"Fetched odds for {sport.value}: {len(data)} events"
        )
        
        return OddsResponseSchema(events=data)
    
    async def get_event_odds(
        self,
        sport: Sport,
        event_id: str,
        regions: Optional[List[OddsRegion]] = None,
        markets: Optional[List[OddsMarket]] = None,
    ) -> EventSchema:
        """
        Get odds for a specific event
        
        Args:
            sport: Sport identifier
            event_id: Event ID
            regions: Bookmaker regions
            markets: Betting markets
        
        Returns:
            EventSchema with event odds
        """
        if regions is None:
            regions = [OddsRegion.EU]
        if markets is None:
            markets = [OddsMarket.H2H]
        
        params = {
            "regions": ",".join([r.value for r in regions]),
            "markets": ",".join([m.value for m in markets]),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        
        endpoint = f"/sports/{sport.value}/events/{event_id}/odds"
        data = await self._make_request(endpoint, params)
        
        logger.info(f"Fetched odds for event {event_id}")
        
        return EventSchema(**data)
    
    async def get_historical_odds(
        self,
        sport: Sport,
        event_id: str,
        regions: Optional[List[OddsRegion]] = None,
        markets: Optional[List[OddsMarket]] = None,
        date: Optional[datetime] = None,
    ) -> EventSchema:
        """
        Get historical odds for an event (if available)
        
        Note: Historical odds may require a paid plan
        
        Args:
            sport: Sport identifier
            event_id: Event ID
            regions: Bookmaker regions
            markets: Betting markets
            date: Historical date
        
        Returns:
            EventSchema with historical odds
        """
        if regions is None:
            regions = [OddsRegion.EU]
        if markets is None:
            markets = [OddsMarket.H2H]
        
        params = {
            "regions": ",".join([r.value for r in regions]),
            "markets": ",".join([m.value for m in markets]),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        
        if date:
            params["date"] = date.isoformat()
        
        endpoint = f"/sports/{sport.value}/events/{event_id}/odds/historical"
        data = await self._make_request(endpoint, params)
        
        logger.info(f"Fetched historical odds for event {event_id}")
        
        return EventSchema(**data)
    
    def get_quota_usage(self) -> Dict[str, Any]:
        """
        Get current API quota usage
        
        Returns:
            Dictionary with quota information
        """
        quota_percentage = (self.requests_used / self.quota_limit * 100) if self.quota_limit > 0 else 0
        
        return {
            "requests_used": self.requests_used,
            "requests_remaining": self.requests_remaining,
            "quota_limit": self.quota_limit,
            "usage_percentage": round(quota_percentage, 2),
            "approaching_limit": quota_percentage >= (self.QUOTA_WARNING_THRESHOLD * 100),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics
        
        Returns:
            Dictionary with request and error counts
        """
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "success_rate": (
                (self.request_count - self.error_count) / self.request_count * 100
                if self.request_count > 0
                else 0
            ),
            **self.get_quota_usage(),
        }
