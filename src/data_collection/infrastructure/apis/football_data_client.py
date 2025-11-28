"""
Football-Data.org API Client

Async HTTP client for Football-Data.org API v4 with:
- Rate limiting (10 requests/minute)
- Exponential backoff retry logic
- Comprehensive error handling
- Pydantic schema validation
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum

import httpx
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.shared.config.settings import settings
from src.shared.logging.logger import get_logger
from src.shared.exceptions.custom_exceptions import (
    APIRateLimitError,
    APIAuthenticationError,
    APINotFoundError,
    APIServerError,
)
from src.data_collection.application.dtos.api_schemas import (
    MatchSchema,
    MatchListSchema,
    TeamSchema,
    H2HSchema,
    StandingsSchema,
)

logger = get_logger(__name__)


class MatchStatus(str, Enum):
    """Match status enumeration"""
    SCHEDULED = "SCHEDULED"
    TIMED = "TIMED"
    IN_PLAY = "IN_PLAY"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    SUSPENDED = "SUSPENDED"
    CANCELLED = "CANCELLED"


class FootballDataClient:
    """
    Async client for Football-Data.org API v4
    
    Features:
    - Rate limiting: 10 requests/minute (free tier)
    - Automatic retries with exponential backoff
    - Request/response logging
    - Error handling for common API errors
    
    Usage:
        async with FootballDataClient() as client:
            matches = await client.get_matches(league_id=2014)
    """
    
    BASE_URL = "https://api.football-data.org/v4"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Football-Data.org API client
        
        Args:
            api_key: API key (defaults to settings.FOOTBALL_DATA_API_KEY)
        """
        self.api_key = api_key or settings.FOOTBALL_DATA_API_KEY
        
        # Rate limiter: 10 requests per minute
        self.rate_limiter = AsyncLimiter(max_rate=10, time_period=60)
        
        # HTTP client configuration
        self.client: Optional[httpx.AsyncClient] = None
        self.headers = {
            "X-Auth-Token": self.api_key,
            "Accept": "application/json",
        }
        
        # Request statistics
        self.request_count = 0
        self.error_count = 0
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=self.headers,
            timeout=30.0,
        )
        logger.info("Football-Data.org API client initialized")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()
        logger.info(
            f"Football-Data.org API client closed. "
            f"Requests: {self.request_count}, Errors: {self.error_count}"
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
        Make rate-limited HTTP request with retry logic
        
        Args:
            endpoint: API endpoint (e.g., "/competitions/2014/matches")
            params: Query parameters
        
        Returns:
            JSON response as dictionary
        
        Raises:
            APIRateLimitError: Rate limit exceeded (429)
            APIAuthenticationError: Invalid API key (401, 403)
            APINotFoundError: Resource not found (404)
            APIServerError: Server error (500+)
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        # Apply rate limiting
        async with self.rate_limiter:
            try:
                logger.debug(f"GET {endpoint} with params {params}")
                
                response = await self.client.get(endpoint, params=params)
                self.request_count += 1
                
                # Handle HTTP errors
                if response.status_code == 429:
                    self.error_count += 1
                    logger.error("Rate limit exceeded")
                    raise APIRateLimitError("Rate limit exceeded. Please wait before retrying.")
                
                elif response.status_code in (401, 403):
                    self.error_count += 1
                    logger.error(f"Authentication failed: {response.status_code}")
                    raise APIAuthenticationError("Invalid API key or insufficient permissions")
                
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
    
    async def get_matches(
        self,
        league_id: int,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        status: Optional[MatchStatus] = None,
    ) -> MatchListSchema:
        """
        Get matches for a specific league/competition
        
        Args:
            league_id: Competition ID (e.g., 2014 for La Liga)
            date_from: Start date filter
            date_to: End date filter
            status: Match status filter
        
        Returns:
            MatchListSchema with list of matches
        
        Example:
            matches = await client.get_matches(
                league_id=2014,
                date_from=datetime.now(),
                date_to=datetime.now() + timedelta(days=7),
                status=MatchStatus.SCHEDULED
            )
        """
        params = {}
        
        if date_from:
            params["dateFrom"] = date_from.strftime("%Y-%m-%d")
        if date_to:
            params["dateTo"] = date_to.strftime("%Y-%m-%d")
        if status:
            params["status"] = status.value
        
        endpoint = f"/competitions/{league_id}/matches"
        data = await self._make_request(endpoint, params)
        
        logger.info(
            f"Fetched {len(data.get('matches', []))} matches for league {league_id}"
        )
        
        return MatchListSchema(**data)
    
    async def get_match_details(self, match_id: int) -> MatchSchema:
        """
        Get detailed information for a specific match
        
        Args:
            match_id: Match ID
        
        Returns:
            MatchSchema with match details
        """
        endpoint = f"/matches/{match_id}"
        data = await self._make_request(endpoint)
        
        logger.info(f"Fetched details for match {match_id}")
        
        return MatchSchema(**data)
    
    async def get_team_stats(self, team_id: int) -> TeamSchema:
        """
        Get team information and statistics
        
        Args:
            team_id: Team ID
        
        Returns:
            TeamSchema with team data
        """
        endpoint = f"/teams/{team_id}"
        data = await self._make_request(endpoint)
        
        logger.info(f"Fetched stats for team {team_id}")
        
        return TeamSchema(**data)
    
    async def get_head_to_head(
        self,
        match_id: int,
        limit: int = 10,
    ) -> H2HSchema:
        """
        Get head-to-head history for teams in a match
        
        Args:
            match_id: Match ID to get H2H for
            limit: Number of previous matches to retrieve
        
        Returns:
            H2HSchema with historical match data
        """
        endpoint = f"/matches/{match_id}/head2head"
        params = {"limit": limit}
        
        data = await self._make_request(endpoint, params)
        
        logger.info(f"Fetched H2H for match {match_id} (limit: {limit})")
        
        return H2HSchema(**data)
    
    async def get_league_standings(self, league_id: int) -> StandingsSchema:
        """
        Get current standings/table for a league
        
        Args:
            league_id: Competition ID
        
        Returns:
            StandingsSchema with league table
        """
        endpoint = f"/competitions/{league_id}/standings"
        data = await self._make_request(endpoint)
        
        logger.info(f"Fetched standings for league {league_id}")
        
        return StandingsSchema(**data)
    
    async def get_upcoming_matches(
        self,
        league_id: int,
        days_ahead: int = 7,
    ) -> MatchListSchema:
        """
        Convenience method to get upcoming matches for next N days
        
        Args:
            league_id: Competition ID
            days_ahead: Number of days to look ahead
        
        Returns:
            MatchListSchema with upcoming matches
        """
        date_from = datetime.now()
        date_to = datetime.now() + timedelta(days=days_ahead)
        
        return await self.get_matches(
            league_id=league_id,
            date_from=date_from,
            date_to=date_to,
            status=MatchStatus.SCHEDULED,
        )
    
    def get_stats(self) -> Dict[str, int]:
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
        }
