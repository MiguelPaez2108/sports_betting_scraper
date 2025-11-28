"""
Football-Data.co.uk Historical Data Scraper

Downloads historical match data from https://www.football-data.co.uk/
Provides 20+ years of match results, odds, and statistics for model training.

Data includes:
- Match results (full-time, half-time)
- Bookmaker odds (Bet365, Pinnacle, etc.)
- Match statistics (shots, corners, cards)
- Multiple leagues (Premier League, La Liga, Serie A, etc.)
"""

import asyncio
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import StringIO

import httpx
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.logging.logger import get_logger
from src.shared.config.settings import settings
from src.data_collection.application.dtos.historical_schemas import (
    HistoricalMatchSchema,
    HistoricalOddsSchema,
)

logger = get_logger(__name__)


class FootballDataUKScraper:
    """
    Scraper for Football-Data.co.uk historical data
    
    Features:
    - Downloads CSV files for multiple seasons
    - Parses match results and odds
    - Supports multiple leagues
    - Stores in database for training
    
    Usage:
        scraper = FootballDataUKScraper()
        await scraper.download_league_history("E0", start_year=2010, end_year=2024)
    """
    
    BASE_URL = "https://www.football-data.co.uk"
    
    # League codes mapping
    LEAGUE_CODES = {
        "premier_league": "E0",
        "championship": "E1",
        "league_one": "E2",
        "league_two": "E3",
        "la_liga": "SP1",
        "la_liga_2": "SP2",
        "serie_a": "I1",
        "serie_b": "I2",
        "bundesliga": "D1",
        "bundesliga_2": "D2",
        "ligue_1": "F1",
        "ligue_2": "F2",
        "eredivisie": "N1",
        "jupiler_league": "B1",
        "primeira_liga": "P1",
        "scottish_premiership": "SC0",
        "turkish_super_lig": "T1",
        "greek_super_league": "G1",
    }
    
    def __init__(self, download_dir: Optional[Path] = None):
        """
        Initialize scraper
        
        Args:
            download_dir: Directory to save downloaded CSV files
        """
        self.download_dir = download_dir or Path(settings.data_path) / "historical"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.client: Optional[httpx.AsyncClient] = None
        
        # Statistics
        self.files_downloaded = 0
        self.matches_parsed = 0
        self.errors = 0
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("Football-Data.co.uk scraper initialized")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()
        logger.info(
            f"Scraper closed. Files: {self.files_downloaded}, "
            f"Matches: {self.matches_parsed}, Errors: {self.errors}"
        )
    
    def _build_csv_url(self, league_code: str, season: str) -> str:
        """
        Build URL for CSV file
        
        Args:
            league_code: League code (e.g., "E0" for Premier League)
            season: Season string (e.g., "2324" for 2023-24)
        
        Returns:
            Full URL to CSV file
        
        Example:
            https://www.football-data.co.uk/mmz4281/2324/E0.csv
        """
        return f"{self.BASE_URL}/mmz4281/{season}/{league_code}.csv"
    
    def _season_to_string(self, year: int) -> str:
        """
        Convert year to season string
        
        Args:
            year: Starting year (e.g., 2023)
        
        Returns:
            Season string (e.g., "2324")
        """
        next_year = (year + 1) % 100
        return f"{year % 100:02d}{next_year:02d}"
    
    async def download_csv(
        self,
        league_code: str,
        season: str,
        save: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Download CSV file for a league/season
        
        Args:
            league_code: League code
            season: Season string
            save: Whether to save CSV to disk
        
        Returns:
            DataFrame with match data, or None if download fails
        """
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        url = self._build_csv_url(league_code, season)
        
        try:
            logger.info(f"Downloading {league_code} season {season}...")
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Parse CSV
            df = pd.read_csv(StringIO(response.text))
            
            # Save to disk if requested
            if save:
                filename = self.download_dir / f"{league_code}_{season}.csv"
                df.to_csv(filename, index=False)
                logger.debug(f"Saved to {filename}")
            
            self.files_downloaded += 1
            self.matches_parsed += len(df)
            
            logger.info(f"Downloaded {len(df)} matches for {league_code} {season}")
            
            return df
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"CSV not found: {url}")
            else:
                logger.error(f"HTTP error downloading {url}: {e}")
            self.errors += 1
            return None
        
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            self.errors += 1
            return None
    
    async def download_league_history(
        self,
        league_code: str,
        start_year: int = 2010,
        end_year: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """
        Download all available seasons for a league
        
        Args:
            league_code: League code (e.g., "E0")
            start_year: Starting year
            end_year: Ending year (defaults to current year)
        
        Returns:
            List of DataFrames, one per season
        """
        if end_year is None:
            end_year = datetime.now().year
        
        dataframes = []
        
        for year in range(start_year, end_year + 1):
            season = self._season_to_string(year)
            df = await self.download_csv(league_code, season)
            
            if df is not None:
                # Add metadata
                df['season'] = f"{year}-{year+1}"
                df['league_code'] = league_code
                dataframes.append(df)
            
            # Be polite - small delay between requests
            await asyncio.sleep(0.5)
        
        logger.info(
            f"Downloaded {len(dataframes)} seasons for {league_code} "
            f"({start_year}-{end_year})"
        )
        
        return dataframes
    
    def parse_match_data(self, df: pd.DataFrame) -> List[HistoricalMatchSchema]:
        """
        Parse DataFrame into structured match data
        
        Args:
            df: DataFrame from CSV
        
        Returns:
            List of HistoricalMatchSchema objects
        """
        matches = []
        
        for _, row in df.iterrows():
            try:
                # Parse date (format varies, try multiple formats)
                date_str = str(row.get('Date', ''))
                match_date = None
                
                for fmt in ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d']:
                    try:
                        match_date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                
                if not match_date:
                    logger.warning(f"Could not parse date: {date_str}")
                    continue
                
                # Extract match data
                match = HistoricalMatchSchema(
                    date=match_date,
                    home_team=str(row.get('HomeTeam', '')),
                    away_team=str(row.get('AwayTeam', '')),
                    
                    # Full-time result
                    fthg=int(row.get('FTHG', 0)),  # Full-time home goals
                    ftag=int(row.get('FTAG', 0)),  # Full-time away goals
                    ftr=str(row.get('FTR', '')),   # Full-time result (H/D/A)
                    
                    # Half-time result
                    hthg=int(row.get('HTHG', 0)) if pd.notna(row.get('HTHG')) else None,
                    htag=int(row.get('HTAG', 0)) if pd.notna(row.get('HTAG')) else None,
                    htr=str(row.get('HTR', '')) if pd.notna(row.get('HTR')) else None,
                    
                    # Statistics
                    hs=int(row.get('HS', 0)) if pd.notna(row.get('HS')) else None,  # Home shots
                    as_=int(row.get('AS', 0)) if pd.notna(row.get('AS')) else None,  # Away shots
                    hst=int(row.get('HST', 0)) if pd.notna(row.get('HST')) else None,  # Home shots on target
                    ast=int(row.get('AST', 0)) if pd.notna(row.get('AST')) else None,  # Away shots on target
                    hc=int(row.get('HC', 0)) if pd.notna(row.get('HC')) else None,  # Home corners
                    ac=int(row.get('AC', 0)) if pd.notna(row.get('AC')) else None,  # Away corners
                    hf=int(row.get('HF', 0)) if pd.notna(row.get('HF')) else None,  # Home fouls
                    af=int(row.get('AF', 0)) if pd.notna(row.get('AF')) else None,  # Away fouls
                    hy=int(row.get('HY', 0)) if pd.notna(row.get('HY')) else None,  # Home yellow cards
                    ay=int(row.get('AY', 0)) if pd.notna(row.get('AY')) else None,  # Away yellow cards
                    hr=int(row.get('HR', 0)) if pd.notna(row.get('HR')) else None,  # Home red cards
                    ar=int(row.get('AR', 0)) if pd.notna(row.get('AR')) else None,  # Away red cards
                    
                    # Bookmaker odds (Bet365 example)
                    b365h=float(row.get('B365H', 0)) if pd.notna(row.get('B365H')) else None,
                    b365d=float(row.get('B365D', 0)) if pd.notna(row.get('B365D')) else None,
                    b365a=float(row.get('B365A', 0)) if pd.notna(row.get('B365A')) else None,
                    
                    # Metadata
                    season=str(row.get('season', '')),
                    league_code=str(row.get('league_code', '')),
                )
                
                matches.append(match)
            
            except Exception as e:
                logger.error(f"Error parsing match row: {e}")
                continue
        
        return matches
    
    async def download_all_major_leagues(
        self,
        start_year: int = 2010,
        end_year: Optional[int] = None,
    ) -> Dict[str, List[pd.DataFrame]]:
        """
        Download history for all major leagues
        
        Args:
            start_year: Starting year
            end_year: Ending year
        
        Returns:
            Dictionary mapping league names to list of DataFrames
        """
        major_leagues = {
            "Premier League": "E0",
            "La Liga": "SP1",
            "Serie A": "I1",
            "Bundesliga": "D1",
            "Ligue 1": "F1",
        }
        
        results = {}
        
        for league_name, league_code in major_leagues.items():
            logger.info(f"Downloading {league_name}...")
            dataframes = await self.download_league_history(
                league_code,
                start_year,
                end_year
            )
            results[league_name] = dataframes
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get scraper statistics"""
        return {
            "files_downloaded": self.files_downloaded,
            "matches_parsed": self.matches_parsed,
            "errors": self.errors,
        }
