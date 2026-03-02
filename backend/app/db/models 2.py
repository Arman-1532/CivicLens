"""
SQLAlchemy ORM models for the CivicLens database.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    """
    Stores registered citizen users for tracking their own complaints.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(200), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(150), nullable=False)
    role = Column(String(20), nullable=False, default="citizen") # citizen | department
    assigned_department = Column(String(150), nullable=True) # Only for role="department"
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # Relationships
    complaints = relationship("Complaint", back_populates="user")
    notifications = relationship("Notification", back_populates="user")

class Complaint(Base):
    """
    Stores every submitted citizen complaint and its AI classification result.
    """
    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Human-readable tracking number e.g. CL-2026-00001
    tracking_number = Column(String(20), unique=True, nullable=False, index=True)

    # Citizen information
    citizen_name  = Column(String(150), nullable=False)
    citizen_email = Column(String(200), nullable=False)
    citizen_phone = Column(String(30),  nullable=True)
    location      = Column(String(250), nullable=False)
    
    # Linked user (optional)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)


    # Complaint content
    complaint_text = Column(Text, nullable=False)

    # AI classification results
    category   = Column(String(100), nullable=False)
    department = Column(String(150), nullable=False)
    urgency    = Column(String(20),  nullable=False)
    confidence = Column(Float,       nullable=False)

    # Lifecycle
    # status: "pending" | "in_progress" | "resolved"
    status           = Column(String(20),  nullable=False, default="pending")
    department_notes = Column(Text,        nullable=True)

    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="complaints")


class Notification(Base):
    """
    Stores notifications for users when their complaint status changes.
    """
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    complaint_id = Column(Integer, ForeignKey("complaints.id"), nullable=False)
    message = Column(String(255), nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="notifications")
    complaint = relationship("Complaint")
