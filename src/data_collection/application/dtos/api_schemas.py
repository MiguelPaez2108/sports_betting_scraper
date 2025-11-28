"""
API Schemas

Pydantic models for API request/response validation.
Ensures type safety and data validation for both Football-Data.org and The Odds API.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, validator


# ============================================================================
# Football-Data.org Schemas
# ============================================================================

class TeamInfoSchema(BaseModel):
    """Team information"""
    id: int
    name: str
    shortName: Optional[str] = None
    tla: Optional[str] = None  # Three-letter abbreviation
    crest: Optional[str] = None  # Team logo URL


class ScoreSchema(BaseModel):
    """Match score"""
    home: Optional[int] = None
    away: Optional[int] = None


class FullScoreSchema(BaseModel):
    """Full match score with halftime"""
    home: Optional[int] = None
    away: Optional[int] = None
    duration: str = "REGULAR"


class MatchScoreSchema(BaseModel):
    """Complete match score breakdown"""
    winner: Optional[str] = None  # HOME_TEAM, AWAY_TEAM, DRAW
    duration: str = "REGULAR"
    fullTime: Optional[ScoreSchema] = None
    halfTime: Optional[ScoreSchema] = None


class CompetitionSchema(BaseModel):
    """Competition/League information"""
    id: int
    name: str
    code: Optional[str] = None
    type: str = "LEAGUE"
    emblem: Optional[str] = None


class MatchSchema(BaseModel):
    """Individual match data"""
    id: int
    utcDate: datetime
    status: str  # SCHEDULED, TIMED, IN_PLAY, FINISHED, etc.
    matchday: Optional[int] = None
    stage: Optional[str] = None
    group: Optional[str] = None
    lastUpdated: datetime
    
    competition: CompetitionSchema
    homeTeam: TeamInfoSchema
    awayTeam: TeamInfoSchema
    score: MatchScoreSchema
    
    # Additional fields
    venue: Optional[str] = None
    referees: List[Dict[str, Any]] = Field(default_factory=list)


class MatchListSchema(BaseModel):
    """List of matches response"""
    filters: Dict[str, Any] = Field(default_factory=dict)
    resultSet: Dict[str, Any] = Field(default_factory=dict)
    competition: Optional[CompetitionSchema] = None
    matches: List[MatchSchema]


class TeamStatsSchema(BaseModel):
    """Team statistics"""
    played: int = 0
    won: int = 0
    draw: int = 0
    lost: int = 0
    points: int = 0
    goalsFor: int = 0
    goalsAgainst: int = 0
    goalDifference: int = 0


class TeamSchema(BaseModel):
    """Complete team data"""
    id: int
    name: str
    shortName: Optional[str] = None
    tla: Optional[str] = None
    crest: Optional[str] = None
    address: Optional[str] = None
    website: Optional[str] = None
    founded: Optional[int] = None
    clubColors: Optional[str] = None
    venue: Optional[str] = None
    
    # Squad and staff
    squad: List[Dict[str, Any]] = Field(default_factory=list)
    staff: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Competition data
    runningCompetitions: List[CompetitionSchema] = Field(default_factory=list)


class H2HSchema(BaseModel):
    """Head-to-head history"""
    aggregates: Dict[str, Any] = Field(default_factory=dict)
    matches: List[MatchSchema]


class TablePositionSchema(BaseModel):
    """League table position"""
    position: int
    team: TeamInfoSchema
    playedGames: int
    form: Optional[str] = None
    won: int
    draw: int
    lost: int
    points: int
    goalsFor: int
    goalsAgainst: int
    goalDifference: int


class StandingSchema(BaseModel):
    """League standing/table"""
    stage: str
    type: str
    group: Optional[str] = None
    table: List[TablePositionSchema]


class StandingsSchema(BaseModel):
    """Complete standings response"""
    filters: Dict[str, Any] = Field(default_factory=dict)
    competition: CompetitionSchema
    season: Dict[str, Any]
    standings: List[StandingSchema]


# ============================================================================
# The Odds API Schemas
# ============================================================================

class OutcomeSchema(BaseModel):
    """Individual betting outcome"""
    name: str  # Team name or outcome (e.g., "Over", "Under")
    price: float  # Odds in decimal format
    
    @validator('price')
    def validate_price(cls, v):
        """Ensure odds are positive"""
        if v <= 1.0:
            raise ValueError("Odds must be greater than 1.0")
        return v


class MarketSchema(BaseModel):
    """Betting market (e.g., h2h, totals)"""
    key: str  # Market identifier (h2h, spreads, totals)
    last_update: datetime
    outcomes: List[OutcomeSchema]


class BookmakerSchema(BaseModel):
    """Bookmaker with odds"""
    key: str  # Bookmaker identifier
    title: str  # Bookmaker name
    last_update: datetime
    markets: List[MarketSchema]


class EventSchema(BaseModel):
    """Betting event (match)"""
    id: str
    sport_key: str
    sport_title: str
    commence_time: datetime
    home_team: str
    away_team: str
    bookmakers: List[BookmakerSchema]


class OddsResponseSchema(BaseModel):
    """Response containing multiple events with odds"""
    events: List[EventSchema]


class SportSchema(BaseModel):
    """Sport information"""
    key: str
    group: str
    title: str
    description: str
    active: bool
    has_outrights: bool


# ============================================================================
# Internal/Combined Schemas
# ============================================================================

class CombinedMatchDataSchema(BaseModel):
    """
    Combined match data from both APIs
    Used internally after data correlation
    """
    # Football-Data.org data
    football_data_match_id: int
    football_data: MatchSchema
    
    # The Odds API data
    odds_api_event_id: Optional[str] = None
    odds_data: Optional[EventSchema] = None
    
    # Correlation metadata
    correlation_confidence: float = Field(ge=0.0, le=1.0)
    correlation_method: str  # "exact_match", "fuzzy_match", "manual"
    last_synced: datetime


class OddsMovementSchema(BaseModel):
    """Track odds movement over time"""
    match_id: int
    bookmaker: str
    market: str
    outcome: str
    
    # Odds at different times
    odds_24h: Optional[float] = None
    odds_12h: Optional[float] = None
    odds_6h: Optional[float] = None
    odds_1h: Optional[float] = None
    odds_current: float
    
    # Movement analysis
    movement_percentage: Optional[float] = None
    movement_direction: Optional[str] = None  # "up", "down", "stable"
    
    timestamp: datetime


class DataQualitySchema(BaseModel):
    """Data quality metrics"""
    source: str  # "football_data", "odds_api"
    endpoint: str
    timestamp: datetime
    
    # Quality indicators
    completeness: float = Field(ge=0.0, le=1.0)  # % of fields populated
    freshness_minutes: int  # How old is the data
    error_count: int = 0
    
    # Validation
    schema_valid: bool = True
    data_anomalies: List[str] = Field(default_factory=list)
