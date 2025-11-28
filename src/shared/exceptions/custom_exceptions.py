"""
Custom exceptions for the betting predictor application.
"""


class BettingPredictorException(Exception):
    """Base exception for all application errors"""
    pass


class APIException(BettingPredictorException):
    """Exception raised for API-related errors"""
    pass


class FootballDataAPIException(APIException):
    """Exception raised for Football-Data.org API errors"""
    pass


class OddsAPIException(APIException):
    """Exception raised for The Odds API errors"""
    pass


class RateLimitException(APIException):
    """Exception raised when API rate limit is exceeded"""
    pass


class CacheException(BettingPredictorException):
    """Exception raised for cache-related errors"""
    pass


class DatabaseException(BettingPredictorException):
    """Exception raised for database errors"""
    pass


class FeatureEngineeringException(BettingPredictorException):
    """Exception raised during feature calculation"""
    pass


class ModelException(BettingPredictorException):
    """Exception raised for ML model errors"""
    pass


class ModelNotTrainedException(ModelException):
    """Exception raised when trying to use untrained model"""
    pass


class PredictionException(BettingPredictorException):
    """Exception raised during prediction"""
    pass


class RecommendationException(BettingPredictorException):
    """Exception raised during recommendation generation"""
    pass


class ValidationException(BettingPredictorException):
    """Exception raised for validation errors"""
    pass


class DataLeakageException(ValidationException):
    """Exception raised when data leakage is detected"""
    pass
