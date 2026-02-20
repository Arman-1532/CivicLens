"""
API package initialization.
"""

from fastapi import APIRouter
from .routes import complaint, health

# Create main API router
api_router = APIRouter()

# Include route modules
api_router.include_router(health.router)
api_router.include_router(complaint.router)

