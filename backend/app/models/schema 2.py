"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator, EmailStr
from typing import Optional, List
from datetime import datetime
from enum import Enum


class CategoryEnum(str, Enum):
    """Complaint categories for classification."""
    CORRUPTION = "Corruption"
    UTILITY_ISSUE = "Utility Issue"
    SERVICE_DELAY = "Service Delay"
    HARASSMENT = "Harassment"
    FINANCIAL_ISSUE = "Financial Issue"
    LAW_ENFORCEMENT = "Law Enforcement Issue"


class UrgencyEnum(str, Enum):
    """Urgency levels for complaints."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class StatusEnum(str, Enum):
    """Status states for a complaint lifecycle."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"


class DepartmentEnum(str, Enum):
    """Government departments for complaint routing."""
    ANTI_CORRUPTION = "Anti-Corruption Bureau"
    PUBLIC_UTILITIES = "Public Utilities Department"
    ADMINISTRATIVE = "Administrative Services"
    WOMENS_COMMISSION = "Women's Commission / HR"
    FINANCE = "Finance Department"
    POLICE = "Police Department / Law Enforcement"


# ---------------------------------------------------------------------------
# User & Auth schemas
# ---------------------------------------------------------------------------

class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: Optional[str] = "citizen"
    assigned_department: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None



# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ComplaintSubmitRequest(BaseModel):
    """Full complaint submission with citizen information."""

    citizen_name: str = Field(
        ..., min_length=2, max_length=150,
        description="Full name of the citizen submitting the complaint",
        examples=["Ali Hassan"]
    )
    citizen_email: str = Field(
        ..., max_length=200,
        description="Contact email of the citizen",
        examples=["ali@example.com"]
    )
    citizen_phone: Optional[str] = Field(
        None, max_length=30,
        description="Optional phone number",
        examples=["01700000000"]
    )
    location: str = Field(
        ..., min_length=3, max_length=250,
        description="Area / address of the complaint",
        examples=["Mirpur, Dhaka"]
    )
    complaint_text: str = Field(
        ..., min_length=10, max_length=10000,
        description="Detailed description of the complaint",
        examples=["The water supply has been irregular for the past two weeks."]
    )
    user_id: Optional[int] = None


    @field_validator('complaint_text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Complaint text cannot be empty")
        return v.strip()

    @field_validator('citizen_name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Citizen name cannot be empty")
        return v.strip()

    @field_validator('location')
    @classmethod
    def validate_location(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Location cannot be empty")
        return v.strip()


# Legacy – kept for the /predict endpoint shortcut
class ComplaintRequest(BaseModel):
    """Schema for quick prediction-only requests (no DB storage)."""

    text: str = Field(
        ..., min_length=10, max_length=10000,
        description="The complaint text to classify"
    )

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Complaint text cannot be empty")
        return v.strip()


class ComplaintStatusUpdate(BaseModel):
    """Schema for department updating a complaint's status."""

    status: StatusEnum = Field(..., description="New status for the complaint")
    notes: Optional[str] = Field(
        None, max_length=2000,
        description="Optional notes from the department"
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Schema for quick prediction-only response (legacy)."""

    category: str
    department: str
    urgency: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "category": "Utility Issue",
                "department": "Public Utilities Department",
                "urgency": "Medium",
                "confidence": 0.87,
                "timestamp": "2026-02-20T10:30:00Z"
            }
        }


class ComplaintRecord(BaseModel):
    """Full complaint record as stored in the database."""

    id: int
    tracking_number: str
    citizen_name: str
    citizen_email: str
    citizen_phone: Optional[str]
    location: str
    complaint_text: str
    category: str
    department: str
    urgency: str
    confidence: float
    status: str
    department_notes: Optional[str]
    user_id: Optional[int]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NotificationRecord(BaseModel):
    """Notification record schema."""
    id: int
    user_id: int
    complaint_id: int
    message: str
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    """Aggregated statistics from the database."""

    total: int
    by_status: dict
    by_urgency: dict
    by_category: dict
    by_department: dict


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    version: str
    model_loaded: bool
    vectorizer_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str
    message: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
