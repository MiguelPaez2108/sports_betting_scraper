"""
Redis Cache Manager

Provides caching functionality with configurable TTLs for different data types.
Supports cache warming, invalidation, and hit/miss tracking.
"""

import json
import hashlib
from typing import Any, Optional, Callable
from functools import wraps
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

from src.shared.logging.logger import get_logger
from src.shared.config.settings import settings
from src.shared.exceptions.custom_exceptions import CacheException, CacheConnectionError

logger = get_logger(__name__)


class RedisCache:
    """
    Redis cache manager with async support
    
    Features:
    - Configurable TTLs per data type
    - Automatic serialization/deserialization
    - Cache hit/miss tracking
    - Cache warming capabilities
    - Namespace support for key organization
    
    Usage:
        cache = RedisCache()
        await cache.set("matches:123", match_data, ttl=3600)
        data = await cache.get("matches:123")
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis cache
        
        Args:
            redis_url: Redis connection URL (defaults to settings)
        """
        self.redis_url = redis_url or settings.redis_url
        self.client: Optional[Redis] = None
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
        # TTL configurations (seconds)
        self.ttls = {
            'matches': settings.cache_ttl_matches,  # 1 hour
            'odds': settings.cache_ttl_odds,  # 5 minutes
            'standings': settings.cache_ttl_standings,  # 6 hours
            'team_stats': 86400,  # 24 hours
            'h2h': 43200,  # 12 hours
            'predictions': 1800,  # 30 minutes
        }
    
    async def connect(self):
        """Establish Redis connection"""
        try:
            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CacheConnectionError(f"Redis connection failed: {e}")
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    def _make_key(self, namespace: str, key: str) -> str:
        """
        Create namespaced cache key
        
        Args:
            namespace: Cache namespace (e.g., 'matches', 'odds')
            key: Specific key
        
        Returns:
            Formatted cache key
        """
        return f"{namespace}:{key}"
    
    def _serialize(self, data: Any) -> str:
        """Serialize data to JSON string"""
        return json.dumps(data, default=str)
    
    def _deserialize(self, data: str) -> Any:
        """Deserialize JSON string to data"""
        return json.loads(data)
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            namespace: Cache namespace
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        if not self.client:
            raise CacheConnectionError("Redis client not connected")
        
        cache_key = self._make_key(namespace, key)
        
        try:
            value = await self.client.get(cache_key)
            
            if value:
                self.hits += 1
                logger.debug(f"Cache HIT: {cache_key}")
                return self._deserialize(value)
            else:
                self.misses += 1
                logger.debug(f"Cache MISS: {cache_key}")
                return None
        
        except Exception as e:
            logger.error(f"Cache get error for {cache_key}: {e}")
            return None
    
    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses namespace default if None)
        
        Returns:
            True if successful
        """
        if not self.client:
            raise CacheConnectionError("Redis client not connected")
        
        cache_key = self._make_key(namespace, key)
        
        # Use namespace TTL if not specified
        if ttl is None:
            ttl = self.ttls.get(namespace, 3600)
        
        try:
            serialized = self._serialize(value)
            await self.client.setex(cache_key, ttl, serialized)
            logger.debug(f"Cache SET: {cache_key} (TTL: {ttl}s)")
            return True
        
        except Exception as e:
            logger.error(f"Cache set error for {cache_key}: {e}")
            return False
    
    async def delete(self, namespace: str, key: str) -> bool:
        """
        Delete value from cache
        
        Args:
            namespace: Cache namespace
            key: Cache key
        
        Returns:
            True if deleted
        """
        if not self.client:
            raise CacheConnectionError("Redis client not connected")
        
        cache_key = self._make_key(namespace, key)
        
        try:
            result = await self.client.delete(cache_key)
            logger.debug(f"Cache DELETE: {cache_key}")
            return result > 0
        
        except Exception as e:
            logger.error(f"Cache delete error for {cache_key}: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern
        
        Args:
            pattern: Key pattern (e.g., 'matches:*')
        
        Returns:
            Number of keys deleted
        """
        if not self.client:
            raise CacheConnectionError("Redis client not connected")
        
        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(f"Deleted {deleted} keys matching '{pattern}'")
                return deleted
            
            return 0
        
        except Exception as e:
            logger.error(f"Cache delete pattern error for '{pattern}': {e}")
            return 0
    
    async def exists(self, namespace: str, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.client:
            raise CacheConnectionError("Redis client not connected")
        
        cache_key = self._make_key(namespace, key)
        return await self.client.exists(cache_key) > 0
    
    async def get_ttl(self, namespace: str, key: str) -> int:
        """Get remaining TTL for key in seconds"""
        if not self.client:
            raise CacheConnectionError("Redis client not connected")
        
        cache_key = self._make_key(namespace, key)
        return await self.client.ttl(cache_key)
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace"""
        pattern = f"{namespace}:*"
        return await self.delete_pattern(pattern)
    
    async def clear_all(self) -> bool:
        """Clear entire cache (use with caution!)"""
        if not self.client:
            raise CacheConnectionError("Redis client not connected")
        
        try:
            await self.client.flushdb()
            logger.warning("Cache cleared (FLUSHDB)")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total,
            'hit_rate': round(hit_rate, 2)
        }


def cache_key_from_args(*args, **kwargs) -> str:
    """
    Generate cache key from function arguments
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Hash of arguments as cache key
    """
    # Create string representation of args
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
    key_str = ":".join(key_parts)
    
    # Hash for consistent key length
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(namespace: str, ttl: Optional[int] = None):
    """
    Decorator for caching function results
    
    Args:
        namespace: Cache namespace
        ttl: Time to live in seconds
    
    Usage:
        @cached('matches', ttl=3600)
        async def get_matches(league_id: int):
            # Expensive operation
            return matches
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache instance
            cache = RedisCache()
            await cache.connect()
            
            try:
                # Generate cache key from arguments
                cache_key = cache_key_from_args(*args, **kwargs)
                
                # Try to get from cache
                cached_value = await cache.get(namespace, cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                await cache.set(namespace, cache_key, result, ttl)
                
                return result
            
            finally:
                await cache.disconnect()
        
        return wrapper
    return decorator


# Global cache instance
_cache_instance: Optional[RedisCache] = None


async def get_cache() -> RedisCache:
    """Get global cache instance"""
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = RedisCache()
        await _cache_instance.connect()
    
    return _cache_instance
