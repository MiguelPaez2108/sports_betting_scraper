"""
Redis connection and caching utilities.
"""
import json
from typing import Any, Optional
import redis.asyncio as redis
from redis.asyncio import Redis

from src.shared.config.settings import settings
from src.shared.logging.logger import logger


class RedisCache:
    """Redis cache manager with async support"""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.redis.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.error(f"Redis connection error: {str(e)}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        if not self.redis:
            await self.connect()
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (None = no expiration)
        """
        if not self.redis:
            await self.connect()
        
        try:
            serialized = json.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {str(e)}")
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if not self.redis:
            await self.connect()
        
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {str(e)}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis:
            await self.connect()
        
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {str(e)}")
            return False
    
    async def clear_pattern(self, pattern: str):
        """Delete all keys matching pattern"""
        if not self.redis:
            await self.connect()
        
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys matching pattern: {pattern}")
        except Exception as e:
            logger.error(f"Redis CLEAR_PATTERN error for pattern {pattern}: {str(e)}")


# Global cache instance
cache = RedisCache()
