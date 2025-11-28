"""
Settings and configuration management using Pydantic Settings.
Loads configuration from environment variables and .env file.
"""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # API Keys
    football_data_api_key: str = Field(..., description="Football-Data.org API key")
    odds_api_key: str = Field(..., description="The Odds API key")
    
    # Database URLs
    database_url: str = Field(..., description="PostgreSQL connection URL")
    mongodb_url: str = Field(..., description="MongoDB connection URL")
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    
    # Environment
    environment: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Security
    secret_key: str = Field(default="change-this-in-production", description="Secret key for JWT")
    allowed_origins: str = Field(default="http://localhost:3000,http://localhost:8000", description="CORS allowed origins")
    
    # ML Configuration
    model_path: str = Field(default="./models", description="Path to ML models")
    data_path: str = Field(default="./data", description="Path to data files")
    feature_store_path: str = Field(default="./data/features", description="Path to feature store")
    
    # League IDs (Football-Data.org competition IDs)
    league_laliga_id: int = Field(default=2014, description="La Liga competition ID")
    league_seriea_id: int = Field(default=2019, description="Serie A competition ID")
    league_premier_id: int = Field(default=2021, description="Premier League competition ID")
    league_bundesliga_id: int = Field(default=2002, description="Bundesliga competition ID")
    league_champions_id: int = Field(default=2001, description="Champions League competition ID")
    
    # Prediction Thresholds
    min_confidence_threshold: float = Field(default=0.70, description="Minimum confidence to recommend bet")
    min_edge_threshold: float = Field(default=0.05, description="Minimum edge over bookmaker")
    max_stake_percentage: float = Field(default=0.05, description="Maximum stake as % of bankroll")
    kelly_fraction: float = Field(default=0.25, description="Kelly Criterion safety factor")
    
    # Cache TTLs (seconds)
    cache_ttl_matches: int = Field(default=3600, description="Cache TTL for matches (1 hour)")
    cache_ttl_odds: int = Field(default=300, description="Cache TTL for odds (5 minutes)")
    cache_ttl_standings: int = Field(default=21600, description="Cache TTL for standings (6 hours)")
    
    # Rate Limiting
    football_data_rate_limit: int = Field(default=10, description="Football-Data API requests per minute")
    odds_api_monthly_limit: int = Field(default=500, description="The Odds API monthly request limit")
    
    # Monitoring
    sentry_dsn: str = Field(default="", description="Sentry DSN for error tracking")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking server")
    
    # Celery
    celery_broker_url: str = Field(default="redis://localhost:6379/0", description="Celery broker URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", description="Celery result backend")
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse allowed origins into list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @property
    def all_league_ids(self) -> List[int]:
        """Get all configured league IDs"""
        return [
            self.league_laliga_id,
            self.league_seriea_id,
            self.league_premier_id,
            self.league_bundesliga_id,
            self.league_champions_id
        ]
    
    @property
    def league_names(self) -> dict:
        """Map league IDs to names"""
        return {
            self.league_laliga_id: "La Liga",
            self.league_seriea_id: "Serie A",
            self.league_premier_id: "Premier League",
            self.league_bundesliga_id: "Bundesliga",
            self.league_champions_id: "Champions League"
        }


# Global settings instance
settings = Settings()
