"""
MongoDB connection and client management using Motor (async MongoDB driver).
"""
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.shared.config.settings import settings
from src.shared.logging.logger import logger


class MongoDB:
    """MongoDB connection manager"""
    
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None
    
    @classmethod
    async def connect(cls):
        """Connect to MongoDB"""
        try:
            cls.client = AsyncIOMotorClient(settings.mongodb_url)
            # Extract database name from URL or use default
            cls.db = cls.client.get_default_database()
            
            # Test connection
            await cls.client.admin.command('ping')
            logger.info("MongoDB connected successfully")
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")
            raise
    
    @classmethod
    async def close(cls):
        """Close MongoDB connection"""
        if cls.client:
            cls.client.close()
            logger.info("MongoDB connection closed")
    
    @classmethod
    def get_collection(cls, collection_name: str):
        """Get MongoDB collection"""
        if cls.db is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return cls.db[collection_name]


# Convenience function for FastAPI dependency
async def get_mongodb() -> AsyncIOMotorDatabase:
    """Get MongoDB database instance"""
    if MongoDB.db is None:
        await MongoDB.connect()
    return MongoDB.db
