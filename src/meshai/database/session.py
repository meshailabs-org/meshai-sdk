"""
Database Session Management for MeshAI

Provides connection management, session handling, and database utilities.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

from sqlalchemy import create_engine, pool, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
import structlog

from ..core.config import MeshConfig

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """
    Database connection and session manager.
    
    Supports both SQLite (development) and PostgreSQL (production).
    Provides connection pooling and async session management.
    """
    
    def __init__(self, config: Optional[MeshConfig] = None):
        from ..core.config import get_config
        self.config = config or get_config()
        
        self._async_engine = None
        self._sync_engine = None
        self._async_session_factory = None
        self._sync_session_factory = None
        
        self._initialize_engines()
    
    def _get_database_url(self, async_mode: bool = True) -> str:
        """Get database URL from configuration"""
        
        # Check for explicit database URL in environment
        db_url = os.getenv("MESHAI_DATABASE_URL")
        if db_url:
            if async_mode and not db_url.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
                # Convert sync URLs to async
                if db_url.startswith("postgresql://"):
                    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
                elif db_url.startswith("sqlite://"):
                    db_url = db_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
            elif not async_mode and db_url.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
                # Convert async URLs to sync
                if db_url.startswith("postgresql+asyncpg://"):
                    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
                elif db_url.startswith("sqlite+aiosqlite://"):
                    db_url = db_url.replace("sqlite+aiosqlite://", "sqlite://", 1)
            return db_url
        
        # Check for PostgreSQL configuration
        pg_host = os.getenv("MESHAI_POSTGRES_HOST")
        if pg_host:
            pg_user = os.getenv("MESHAI_POSTGRES_USER", "meshai")
            pg_password = os.getenv("MESHAI_POSTGRES_PASSWORD", "meshai")
            pg_database = os.getenv("MESHAI_POSTGRES_DATABASE", "meshai")
            pg_port = os.getenv("MESHAI_POSTGRES_PORT", "5432")
            
            if async_mode:
                return f"postgresql+asyncpg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
            else:
                return f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
        
        # Default to SQLite
        db_path = os.getenv("MESHAI_SQLITE_PATH", "meshai.db")
        
        if async_mode:
            return f"sqlite+aiosqlite:///{db_path}"
        else:
            return f"sqlite:///{db_path}"
    
    def _initialize_engines(self):
        """Initialize database engines with PostgreSQL fallback to SQLite"""
        
        # Try PostgreSQL first, fallback to SQLite
        async_url = self._get_database_url(async_mode=True)
        sync_url = self._get_database_url(async_mode=False)
        
        # Try PostgreSQL connection first, but don't test it during startup
        if async_url.startswith("postgresql"):
            try:
                # PostgreSQL configuration (no connection test during startup)
                connect_args = {
                    "server_settings": {"jit": "off"},
                    "command_timeout": 60
                }
                
                self._async_engine = create_async_engine(
                    async_url,
                    pool_pre_ping=True,
                    echo=self.config.debug_mode,
                    future=True,
                    pool_recycle=3600,
                    connect_args=connect_args
                )
                
                self._sync_engine = create_engine(
                    sync_url,
                    pool_pre_ping=True,
                    echo=self.config.debug_mode,
                    future=True,
                    pool_recycle=3600
                )
                
                logger.info("PostgreSQL engines initialized (connection will be tested on first use)")
                
            except Exception as e:
                logger.warning(f"PostgreSQL engine initialization failed: {e}. Falling back to SQLite.")
                # Fall back to SQLite
                sqlite_path = os.getenv("MESHAI_SQLITE_PATH", "/tmp/meshai.db")
                async_url = f"sqlite+aiosqlite:///{sqlite_path}"
                sync_url = f"sqlite:///{sqlite_path}"
                
                self._async_engine = create_async_engine(
                    async_url,
                    echo=self.config.debug_mode,
                    future=True
                )
                self._sync_engine = create_engine(
                    sync_url,
                    echo=self.config.debug_mode,
                    future=True
                )
                logger.info(f"Using SQLite fallback: {sqlite_path}")
        else:
            # SQLite configuration  
            self._async_engine = create_async_engine(
                async_url,
                echo=self.config.debug_mode,
                future=True
            )
            self._sync_engine = create_engine(
                sync_url,
                echo=self.config.debug_mode,
                future=True
            )
            logger.info("Using SQLite configuration")
        
        # Session factories
        self._async_session_factory = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        self._sync_session_factory = sessionmaker(
            bind=self._sync_engine,
            expire_on_commit=False
        )
        
        logger.info(f"Database engines initialized", 
                   database_type="postgresql" if async_url.startswith("postgresql") else "sqlite")
    
    @property
    def async_engine(self):
        """Get async database engine"""
        return self._async_engine
    
    @property
    def sync_engine(self):
        """Get sync database engine"""
        return self._sync_engine
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_sync_session(self):
        """Get sync database session"""
        return self._sync_session_factory()
    
    async def close(self):
        """Close database connections"""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()
        
        logger.info("Database connections closed")
    
    async def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            async with self.get_async_session() as session:
                # Simple connectivity test
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Convenient function to get database session"""
    db_manager = get_database_manager()
    async with db_manager.get_async_session() as session:
        yield session


# Dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session"""
    async with get_session() as session:
        yield session