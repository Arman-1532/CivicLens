"""
CRUD operations for the Complaint model.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Complaint, User, Notification
from ..core.security import get_password_hash

logger = logging.getLogger(__name__)


async def generate_tracking_number(db: AsyncSession) -> str:
    """Generate a sequential human-readable tracking number like CL-2026-00042."""
    year = datetime.now(timezone.utc).year
    result = await db.execute(select(func.count()).select_from(Complaint))
    count = result.scalar() or 0
    return f"CL-{year}-{count + 1:05d}"


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Retrieve a user by their email."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user(db: AsyncSession, user_id: int) -> Optional[User]:
    """Retrieve a user by their primary key."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    *,
    email: str,
    password: str,
    full_name: str,
    role: str = "citizen",
    assigned_department: Optional[str] = None
) -> User:
    """Create a new user with hashed password."""
    hashed_password = get_password_hash(password)
    user = User(
        email=email,
        hashed_password=hashed_password,
        full_name=full_name,
        role=role,
        assigned_department=assigned_department
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user



async def create_complaint(
    db: AsyncSession,
    *,
    citizen_name: str,
    citizen_email: str,
    citizen_phone: Optional[str],
    location: str,
    complaint_text: str,
    category: str,
    department: str,
    urgency: str,
    confidence: float,
    user_id: Optional[int] = None,
) -> Complaint:
    """Persist a new complaint to the database and return the saved record."""
    tracking_number = await generate_tracking_number(db)

    complaint = Complaint(
        tracking_number=tracking_number,
        citizen_name=citizen_name,
        citizen_email=citizen_email,
        citizen_phone=citizen_phone,
        location=location,
        complaint_text=complaint_text,
        category=category,
        department=department,
        urgency=urgency,
        confidence=confidence,
        user_id=user_id,
        status="pending",
    )

    db.add(complaint)
    await db.commit()
    await db.refresh(complaint)
    logger.info(f"Complaint saved: {tracking_number} – {category} – {department}")
    return complaint


async def get_complaint(db: AsyncSession, complaint_id: int) -> Optional[Complaint]:
    """Retrieve a single complaint by its primary key."""
    result = await db.execute(select(Complaint).where(Complaint.id == complaint_id))
    return result.scalar_one_or_none()


async def get_complaint_by_tracking(db: AsyncSession, tracking_number: str) -> Optional[Complaint]:
    """Retrieve a complaint by its tracking number."""
    result = await db.execute(
        select(Complaint).where(Complaint.tracking_number == tracking_number)
    )
    return result.scalar_one_or_none()


async def get_complaints(
    db: AsyncSession,
    *,
    department: Optional[str] = None,
    status: Optional[str] = None,
    urgency: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[Complaint]:
    """List complaints with optional filters, newest first."""
    stmt = select(Complaint).order_by(Complaint.created_at.desc()).offset(skip).limit(limit)

    if department:
        stmt = stmt.where(Complaint.department == department)
    if status:
        stmt = stmt.where(Complaint.status == status)
    if urgency:
        stmt = stmt.where(Complaint.urgency == urgency)
    
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_user_complaints(db: AsyncSession, user_id: int) -> List[Complaint]:
    """Retrieve all complaints for a specific user."""
    result = await db.execute(
        select(Complaint).where(Complaint.user_id == user_id).order_by(Complaint.created_at.desc())
    )
    return result.scalars().all()



async def update_complaint_status(
    db: AsyncSession,
    complaint_id: int,
    *,
    status: str,
    notes: Optional[str] = None,
) -> Optional[Complaint]:
    """Update the status and optional notes of a complaint."""
    complaint = await get_complaint(db, complaint_id)
    if not complaint:
        return None

    complaint.status = status
    if notes is not None:
        complaint.department_notes = notes
    complaint.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(complaint)

    # Create notification if user is linked
    if complaint.user_id:
        msg = f"Your complaint {complaint.tracking_number} status updated to {status}."
        notification = Notification(
            user_id=complaint.user_id,
            complaint_id=complaint.id,
            message=msg
        )
        db.add(notification)
        await db.commit()

    return complaint


async def get_user_notifications(db: AsyncSession, user_id: int) -> List[Notification]:
    """Retrieve unread notifications for a user."""
    result = await db.execute(
        select(Notification)
        .where(Notification.user_id == user_id)
        .order_by(Notification.created_at.desc())
    )
    return result.scalars().all()


async def mark_notification_read(db: AsyncSession, notification_id: int) -> bool:
    """Mark a notification as read."""
    result = await db.execute(select(Notification).where(Notification.id == notification_id))
    notification = result.scalar_one_or_none()
    if notification:
        notification.is_read = True
        await db.commit()
        return True
    return False



async def get_stats(db: AsyncSession) -> dict:
    """Return aggregate statistics for the dashboard."""
    total_result = await db.execute(select(func.count()).select_from(Complaint))
    total = total_result.scalar() or 0

    # By status
    by_status = {"pending": 0, "in_progress": 0, "resolved": 0}
    for status_val in by_status.keys():
        r = await db.execute(
            select(func.count()).select_from(Complaint).where(Complaint.status == status_val)
        )
        by_status[status_val] = r.scalar() or 0

    # By urgency
    by_urgency = {}
    for urgency_val in ["High", "Medium", "Low"]:
        r = await db.execute(
            select(func.count()).select_from(Complaint).where(Complaint.urgency == urgency_val)
        )
        by_urgency[urgency_val] = r.scalar() or 0

    # By category
    from ..models.complaint_model import VALID_CATEGORIES
    by_category = {}
    for cat in VALID_CATEGORIES:
        r = await db.execute(
            select(func.count()).select_from(Complaint).where(Complaint.category == cat)
        )
        by_category[cat] = r.scalar() or 0

    # By department
    from ..models.complaint_model import CATEGORY_TO_DEPARTMENT
    departments = list(set(CATEGORY_TO_DEPARTMENT.values()))
    by_department = {}
    for dept in departments:
        r = await db.execute(
            select(func.count()).select_from(Complaint).where(Complaint.department == dept)
        )
        by_department[dept] = r.scalar() or 0

    return {
        "total": total,
        "by_status": by_status,
        "by_urgency": by_urgency,
        "by_category": by_category,
        "by_department": by_department,
    }
