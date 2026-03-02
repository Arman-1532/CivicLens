"""
Security utilities for the application.
Placeholder for authentication, rate limiting, etc.
"""

from functools import wraps
from datetime import datetime, timedelta, timezone
from typing import Any, Union, Optional, Callable
import logging
from jose import jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from ..db.database import get_db
from .config import settings
from ..db.models import User

logger = logging.getLogger(__name__)

# Configuration - moved to settings
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

# Using direct bcrypt to avoid passlib + Python 3.12 compatibility issues
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token", auto_error=False)



def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """
    Rate limiting decorator (placeholder implementation).
    In production, use Redis or similar for distributed rate limiting.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Placeholder - implement actual rate limiting logic
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    """
    if not isinstance(text, str):
        return ""

    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '{', '}', '|', '\\', '^', '~', '[', ']', '`']
    sanitized = text
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')

    # Limit length
    max_length = 10000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized.strip()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check if the plain password matches the hashed version."""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'), 
        hashed_password.encode('utf-8')
    )


def get_password_hash(password: str) -> str:
    """Generate a bcrypt hash of the given password."""
    # Ensure password is a string and encode it to bytes
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    # Return as string for database storage
    return hashed.decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user_id(token: str = Depends(oauth2_scheme)) -> Optional[int]:
    """
    Dependency to validate JWT and return the user_id.
    Returns None if no token is provided (allows optional auth).
    """
    if not token:
        return None
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return int(user_id)
    except (jwt.JWTError, ValidationError, ValueError):
        raise credentials_exception


async def get_current_user(
    db: AsyncSession = Depends(get_db),
    user_id: Optional[int] = Depends(get_current_user_id)
) -> Optional[User]:
    """Dependency to get the current user object from the database."""
    if not user_id:
        return None
    from ..db import crud
    user = await crud.get_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user

