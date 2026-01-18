from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import get_settings


def _database_url() -> str:
    settings = get_settings()
    return (
        "postgresql+asyncpg://"
        f"{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


def get_async_engine():
    return create_async_engine(_database_url(), pool_pre_ping=True)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=get_async_engine(), expire_on_commit=False)
