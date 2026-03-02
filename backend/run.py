#!/usr/bin/env python3
"""
Script to run the CivicLens FastAPI server.
"""

import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"Health Check: http://{settings.HOST}:{settings.PORT}/api/v1/health")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

