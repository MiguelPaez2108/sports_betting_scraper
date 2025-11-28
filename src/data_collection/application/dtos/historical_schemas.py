"""
Historical Data Schemas

Pydantic models for historical match data from Football-Data.co.uk
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class HistoricalMatchSchema(BaseModel):
    """Historical match data from Football-Data.co.uk CSV files"""
    
    # Basic match info
    date: datetime
    home_team: str
    away_team: str
    season: str
    league_code: str
    
    # Full-time result
    fthg: int = Field(..., description="Full-time home goals")
    ftag: int = Field(..., description="Full-time away goals")
    ftr: str = Field(..., description="Full-time result (H/D/A)")
    
    # Half-time result
    hthg: Optional[int] = Field(None, description="Half-time home goals")
    htag: Optional[int] = Field(None, description="Half-time away goals")
    htr: Optional[str] = Field(None, description="Half-time result (H/D/A)")
    
    # Match statistics
    hs: Optional[int] = Field(None, description="Home shots")
    as_: Optional[int] = Field(None, description="Away shots", alias="as")
    hst: Optional[int] = Field(None, description="Home shots on target")
    ast: Optional[int] = Field(None, description="Away shots on target")
    hc: Optional[int] = Field(None, description="Home corners")
    ac: Optional[int] = Field(None, description="Away corners")
    hf: Optional[int] = Field(None, description="Home fouls")
    af: Optional[int] = Field(None, description="Away fouls")
    hy: Optional[int] = Field(None, description="Home yellow cards")
    ay: Optional[int] = Field(None, description="Away yellow cards")
    hr: Optional[int] = Field(None, description="Home red cards")
    ar: Optional[int] = Field(None, description="Away red cards")
    
    # Bet365 odds
    b365h: Optional[float] = Field(None, description="Bet365 home win odds")
    b365d: Optional[float] = Field(None, description="Bet365 draw odds")
    b365a: Optional[float] = Field(None, description="Bet365 away win odds")
    
    # Pinnacle odds
    psh: Optional[float] = Field(None, description="Pinnacle home win odds")
    psd: Optional[float] = Field(None, description="Pinnacle draw odds")
    psa: Optional[float] = Field(None, description="Pinnacle away win odds")
    
    # Over/Under 2.5 goals odds
    b365_over_2_5: Optional[float] = Field(None, description="Bet365 over 2.5 goals odds", alias="B365>2.5")
    b365_under_2_5: Optional[float] = Field(None, description="Bet365 under 2.5 goals odds", alias="B365<2.5")
    
    class Config:
        populate_by_name = True


class HistoricalOddsSchema(BaseModel):
    """Historical odds data"""
    match_date: datetime
    home_team: str
    away_team: str
    bookmaker: str
    
    # 1X2 odds
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None
    
    # Over/Under
    over_2_5_odds: Optional[float] = None
    under_2_5_odds: Optional[float] = None
    
    # Asian Handicap
    asian_handicap_home: Optional[float] = None
    asian_handicap_away: Optional[float] = None
    
    # Metadata
    season: str
    league_code: str


class HistoricalDataSummarySchema(BaseModel):
    """Summary of downloaded historical data"""
    league_name: str
    league_code: str
    seasons_downloaded: int
    total_matches: int
    date_range_start: datetime
    date_range_end: datetime
    has_odds_data: bool
    has_statistics: bool
