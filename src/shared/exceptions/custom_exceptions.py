"""
Custom Exceptions

Application-specific exceptions for better error handling and debugging.
"""


class BettingPredictorException(Exception):
    """Base exception for all application errors"""
    pass


# ============================================================================
# API Exceptions
# ============================================================================

class APIException(BettingPredictorException):
    """Base exception for API-related errors"""
    pass


class APIRateLimitError(APIException):
    """Raised when API rate limit is exceeded"""
    pass


class APIQuotaExceededError(APIException):
    """Raised when API quota/monthly limit is exceeded"""
    pass


class APIAuthenticationError(APIException):
    """Raised when API authentication fails (invalid key)"""
    pass


class APINotFoundError(APIException):
    """Raised when API resource is not found (404)"""
    pass


class APIServerError(APIException):
    """Raised when API server returns 5xx error"""
    pass


class APITimeoutError(APIException):
    """Raised when API request times out"""
    pass


# ============================================================================
# Data Exceptions
# ============================================================================

class DataException(BettingPredictorException):
    """Base exception for data-related errors"""
    pass


class DataValidationError(DataException):
    """Raised when data validation fails"""
    pass


class DataCorrelationError(DataException):
    """Raised when data from different sources cannot be correlated"""
    pass


class DataNotFoundError(DataException):
    """Raised when expected data is not found in database"""
    pass


class DataIntegrityError(DataException):
    """Raised when data integrity constraints are violated"""
    pass


# ============================================================================
# Cache Exceptions
# ============================================================================

class CacheException(BettingPredictorException):
    """Base exception for cache-related errors"""
    pass


class CacheConnectionError(CacheException):
    """Raised when cannot connect to cache (Redis)"""
    pass


class CacheKeyError(CacheException):
    """Raised when cache key is invalid or not found"""
    pass


# ============================================================================
# Database Exceptions
# ============================================================================

class DatabaseException(BettingPredictorException):
    """Base exception for database-related errors"""
    pass


class DatabaseConnectionError(DatabaseException):
    """Raised when cannot connect to database"""
    pass


class DatabaseQueryError(DatabaseException):
    """Raised when database query fails"""
    pass


# ============================================================================
# Model Exceptions
# ============================================================================

class ModelException(BettingPredictorException):
    """Base exception for ML model-related errors"""
    pass


class ModelNotFoundError(ModelException):
    """Raised when ML model file is not found"""
    pass


class ModelPredictionError(ModelException):
    """Raised when model prediction fails"""
    pass


class ModelTrainingError(ModelException):
    """Raised when model training fails"""
    pass


# ============================================================================
# Feature Engineering Exceptions
# ============================================================================

class FeatureException(BettingPredictorException):
    """Base exception for feature engineering errors"""
    pass


class FeatureCalculationError(FeatureException):
    """Raised when feature calculation fails"""
    pass


class InsufficientDataError(FeatureException):
    """Raised when not enough data to calculate features"""
    pass


# ============================================================================
# Recommendation Exceptions
# ============================================================================

class RecommendationException(BettingPredictorException):
    """Base exception for recommendation system errors"""
    pass


class NoValueBetsFoundError(RecommendationException):
    """Raised when no value bets are found"""
    pass


class InvalidStakeError(RecommendationException):
    """Raised when stake calculation is invalid"""
    pass
