from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    app_name: str = Field(default="agent-backend", alias="APP_NAME")
    api_prefix: str = Field(default="/api", alias="API_PREFIX")

    postgres_db: str = Field(default="agent_db", alias="POSTGRES_DB")
    postgres_user: str = Field(default="agent_user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="agent_password", alias="POSTGRES_PASSWORD")
    postgres_host: str = Field(default="postgres", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")

    redis_host: str = Field(default="redis", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")

    minio_root_user: str = Field(default="minioadmin", alias="MINIO_ROOT_USER")
    minio_root_password: str = Field(default="minioadmin", alias="MINIO_ROOT_PASSWORD")
    minio_endpoint: str = Field(default="minio:9000", alias="MINIO_ENDPOINT")
    minio_bucket: str = Field(default="agent-objects", alias="MINIO_BUCKET")

    ollama_host: str = Field(default="http://ollama:11434", alias="OLLAMA_HOST")

    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    frontend_port: int = Field(default=3000, alias="FRONTEND_PORT")

    jwt_secret: str = Field(default="change-me", alias="JWT_SECRET")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, alias="REFRESH_TOKEN_EXPIRE_DAYS")

    frontend_url: str = Field(default="http://localhost:3000", alias="FRONTEND_URL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
