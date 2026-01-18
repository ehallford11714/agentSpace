from fastapi import APIRouter

from backend.app.api.routes import health

router = APIRouter()
router.include_router(health.router, tags=["health"])
