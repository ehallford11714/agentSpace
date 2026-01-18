from functools import lru_cache

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "agent-backend"
    api_prefix: str = "/api"


@lru_cache
def get_settings() -> Settings:
    return Settings()
