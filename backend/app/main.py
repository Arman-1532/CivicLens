"""
CivicLens - AI-based Complaint Classification System
FastAPI Application Entry Point
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime

from .core.config import settings
from .api import api_router
from .services.prediction_service import get_prediction_service
from .utils import setup_logging
from .db.database import init_db

# Setup logging
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Initialise SQLite database (creates tables if they don't exist)
    await init_db()
    logger.info("Database initialised")

    # Seed Department accounts
    from .db.database import get_db
    from .db import crud
    from .models.schema import DepartmentEnum
    async for db in get_db():
        for dept in DepartmentEnum:
            # Create internal email like police_dept@civiclens.internal
            email = f"{dept.value.lower().replace(' ', '_').replace('/', '_')}@civiclens.internal"
            existing = await crud.get_user_by_email(db, email)
            if not existing:
                await crud.create_user(
                    db,
                    email=email,
                    password="1234",
                    full_name=f"{dept.value} Official",
                    role="department",
                    assigned_department=dept.value
                )
        break # Only need one session
    logger.info("Department accounts seeded")

    # Load ML models
    prediction_service = get_prediction_service()
    models_loaded = prediction_service.load_models()

    if models_loaded:
        logger.info("ML models loaded successfully")
    else:
        logger.warning("ML models not loaded - prediction service will be unavailable")
        logger.warning("Please train the model first using the ML training pipeline")

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## AI-based Complaint Classification System for E-Governance
    
    This API provides intelligent classification of citizen complaints into categories,
    assigns urgency levels, and routes them to appropriate government departments.
    
    ### Features:
    - **Complaint Classification**: Automatically categorize complaints into 6 categories
    - **Department Routing**: Map complaints to responsible government departments
    - **Urgency Assessment**: Determine complaint urgency (High/Medium/Low)
    - **Confidence Scoring**: Get prediction confidence scores
    
    ### Categories:
    - Corruption
    - Utility Issue
    - Service Delay
    - Harassment
    - Financial Issue
    - Law Enforcement Issue
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with custom response."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Request validation failed",
            "details": errors,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Include API routes
app.include_router(api_router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "AI-based Complaint Classification System for E-Governance",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Direct predict endpoint (convenience alias)
@app.post("/predict", tags=["Shortcuts"])
async def predict_shortcut(request: dict):
    """
    Shortcut endpoint for quick predictions.
    Redirects to /api/v1/complaints/predict
    """
    from .models.schema import ComplaintRequest
    from .api.routes.complaint import predict_complaint

    complaint_request = ComplaintRequest(text=request.get("text", ""))
    return await predict_complaint(complaint_request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

