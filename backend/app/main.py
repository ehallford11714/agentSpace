from fastapi import FastAPI

from backend.app.api.router import router as api_router
from backend.app.core.config import get_settings
from backend.app.exceptions import AppException, app_exception_handler
from backend.app.middleware.request_id import add_request_id


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.middleware("http")(add_request_id)
    app.add_exception_handler(AppException, app_exception_handler)
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
