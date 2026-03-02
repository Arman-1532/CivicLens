"""
Utility functions for the application.
"""

import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


def format_response(data: Dict[str, Any], success: bool = True) -> Dict[str, Any]:
    """
    Format API response with consistent structure.

    Args:
        data: Response data
        success: Whether the operation was successful

    Returns:
        Formatted response dictionary
    """
    return {
        "success": success,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }


def setup_logging(level: str = "INFO") -> None:
    """
    Setup application logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

