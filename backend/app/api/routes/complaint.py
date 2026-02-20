"""
Complaint routes – handles submission, listing, status updates, and statistics.
All submitted complaints are persisted to the SQLite database.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
import logging
from typing import Optional, List

from ...models.schema import (
    ComplaintRequest,
    ComplaintSubmitRequest,
    ComplaintRecord,
    ComplaintStatusUpdate,
    PredictionResponse,
    StatsResponse,
    ErrorResponse,
)
from ...services.prediction_service import get_prediction_service
from ...core.security import sanitize_input
from ...db.database import get_db
from ...db import crud
from ...models.complaint_model import VALID_CATEGORIES, CATEGORY_TO_DEPARTMENT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/complaints", tags=["Complaints"])


# ---------------------------------------------------------------------------
# POST /submit  – main submission endpoint (stores in DB)
# ---------------------------------------------------------------------------

@router.post(
    "/submit",
    response_model=ComplaintRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Submit Complaint",
    description="Submit a complaint with citizen information. AI classifies it and stores it in the database.",
)
async def submit_complaint(
    request: ComplaintSubmitRequest,
    db: AsyncSession = Depends(get_db),
) -> ComplaintRecord:
    """
    Full complaint submission:
    1. Sanitise text
    2. Run AI classification
    3. Persist to database
    4. Return record with tracking number
    """
    prediction_service = get_prediction_service()

    # --- AI classification -------------------------------------------------
    if prediction_service.is_ready:
        try:
            sanitized_text = sanitize_input(request.complaint_text)
            if not sanitized_text or len(sanitized_text) < 10:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Complaint text is too short or invalid after sanitization.",
                )
            result = prediction_service.predict(sanitized_text)
            category = result["category"]
            department = result["department"]
            urgency = result["urgency"]
            confidence = result["confidence"]
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    else:
        # Fallback when model not yet trained: use a placeholder classification
        logger.warning("ML model not loaded – using fallback classification")
        category = "Service Delay"
        department = CATEGORY_TO_DEPARTMENT.get(category, "Administrative Services")
        urgency = "Medium"
        confidence = 0.0

    # --- Persist to database -----------------------------------------------
    complaint = await crud.create_complaint(
        db,
        citizen_name=request.citizen_name,
        citizen_email=request.citizen_email,
        citizen_phone=request.citizen_phone,
        location=request.location,
        complaint_text=request.complaint_text,
        category=category,
        department=department,
        urgency=urgency,
        confidence=confidence,
    )

    logger.info(
        f"Complaint {complaint.tracking_number} saved: {category} → {department} [{urgency}]"
    )
    return complaint


# ---------------------------------------------------------------------------
# GET /  – list all complaints (with optional filters)
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=List[ComplaintRecord],
    status_code=status.HTTP_200_OK,
    summary="List Complaints",
    description="Retrieve complaints with optional filters for department, status, and urgency.",
)
async def list_complaints(
    department: Optional[str] = Query(None, description="Filter by department name"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status: pending | in_progress | resolved"),
    urgency: Optional[str] = Query(None, description="Filter by urgency: High | Medium | Low"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> List[ComplaintRecord]:
    complaints = await crud.get_complaints(
        db,
        department=department,
        status=status_filter,
        urgency=urgency,
        skip=skip,
        limit=limit,
    )
    return complaints


# ---------------------------------------------------------------------------
# GET /stats  – dashboard statistics
# ---------------------------------------------------------------------------

@router.get(
    "/stats",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Statistics",
    description="Retrieve aggregated statistics for the admin dashboard.",
)
async def get_stats(db: AsyncSession = Depends(get_db)) -> StatsResponse:
    stats = await crud.get_stats(db)
    return StatsResponse(**stats)


# ---------------------------------------------------------------------------
# GET /{id}  – single complaint detail
# ---------------------------------------------------------------------------

@router.get(
    "/{complaint_id}",
    response_model=ComplaintRecord,
    status_code=status.HTTP_200_OK,
    summary="Get Complaint",
    description="Retrieve a single complaint by its database ID.",
)
async def get_complaint(
    complaint_id: int,
    db: AsyncSession = Depends(get_db),
) -> ComplaintRecord:
    complaint = await crud.get_complaint(db, complaint_id)
    if not complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")
    return complaint


# ---------------------------------------------------------------------------
# PATCH /{id}/status  – department updates status
# ---------------------------------------------------------------------------

@router.patch(
    "/{complaint_id}/status",
    response_model=ComplaintRecord,
    status_code=status.HTTP_200_OK,
    summary="Update Status",
    description="Department updates the status (pending → in_progress → resolved) and optionally adds notes.",
)
async def update_status(
    complaint_id: int,
    update: ComplaintStatusUpdate,
    db: AsyncSession = Depends(get_db),
) -> ComplaintRecord:
    complaint = await crud.update_complaint_status(
        db,
        complaint_id,
        status=update.status.value,
        notes=update.notes,
    )
    if not complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")
    logger.info(f"Complaint {complaint.tracking_number} status → {update.status.value}")
    return complaint


# ---------------------------------------------------------------------------
# Legacy endpoints kept for backward compatibility
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Quick Classify (no DB storage)",
    description="Classify complaint text only – does NOT store in database. Use /submit instead.",
)
async def predict_complaint(request: ComplaintRequest) -> PredictionResponse:
    """Legacy prediction-only endpoint."""
    prediction_service = get_prediction_service()

    if not prediction_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not available. Models are not loaded.",
        )

    try:
        sanitized_text = sanitize_input(request.text)
        if not sanitized_text or len(sanitized_text) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Complaint text is too short or invalid after sanitization.",
            )
        result = prediction_service.predict(sanitized_text)
        return PredictionResponse(
            category=result["category"],
            department=result["department"],
            urgency=result["urgency"],
            confidence=result["confidence"],
            timestamp=datetime.now(timezone.utc),
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/categories",
    status_code=status.HTTP_200_OK,
    summary="Get Categories",
    description="Get list of all complaint categories.",
)
async def get_categories():
    return {
        "categories": [
            {"name": cat, "department": CATEGORY_TO_DEPARTMENT.get(cat, "Administrative Services")}
            for cat in VALID_CATEGORIES
        ]
    }


@router.get(
    "/model-info",
    status_code=status.HTTP_200_OK,
    summary="Get Model Info",
    description="Get information about the loaded ML model.",
)
async def get_model_info():
    prediction_service = get_prediction_service()
    return prediction_service.get_model_info()
