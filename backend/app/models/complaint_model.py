"""
Complaint model definitions and mappings.
Contains category to department and urgency mappings.
"""

from typing import Dict, List
from .schema import CategoryEnum, DepartmentEnum, UrgencyEnum


# Mapping of categories to responsible departments
CATEGORY_TO_DEPARTMENT: Dict[str, str] = {
    CategoryEnum.CORRUPTION.value: DepartmentEnum.ANTI_CORRUPTION.value,
    CategoryEnum.UTILITY_ISSUE.value: DepartmentEnum.PUBLIC_UTILITIES.value,
    CategoryEnum.SERVICE_DELAY.value: DepartmentEnum.ADMINISTRATIVE.value,
    CategoryEnum.HARASSMENT.value: DepartmentEnum.WOMENS_COMMISSION.value,
    CategoryEnum.FINANCIAL_ISSUE.value: DepartmentEnum.FINANCE.value,
    CategoryEnum.LAW_ENFORCEMENT.value: DepartmentEnum.POLICE.value,
}

# Keywords that indicate high urgency
HIGH_URGENCY_KEYWORDS: List[str] = [
    "urgent", "emergency", "immediate", "critical", "danger",
    "threat", "violence", "assault", "death", "dying", "severe",
    "life-threatening", "attack", "abuse", "criminal", "illegal",
    "bribe", "extortion", "harassment", "discriminat"
]

# Keywords that indicate medium urgency
MEDIUM_URGENCY_KEYWORDS: List[str] = [
    "delay", "waiting", "pending", "overdue", "problem",
    "issue", "complaint", "concern", "failure", "broken",
    "damaged", "not working", "poor", "bad", "incorrect"
]

# Category-based default urgency (if no keywords match)
CATEGORY_DEFAULT_URGENCY: Dict[str, str] = {
    CategoryEnum.CORRUPTION.value: UrgencyEnum.HIGH.value,
    CategoryEnum.HARASSMENT.value: UrgencyEnum.HIGH.value,
    CategoryEnum.LAW_ENFORCEMENT.value: UrgencyEnum.HIGH.value,
    CategoryEnum.FINANCIAL_ISSUE.value: UrgencyEnum.MEDIUM.value,
    CategoryEnum.UTILITY_ISSUE.value: UrgencyEnum.MEDIUM.value,
    CategoryEnum.SERVICE_DELAY.value: UrgencyEnum.LOW.value,
}


def get_department_for_category(category: str) -> str:
    """
    Get the responsible department for a given category.

    Args:
        category: The complaint category

    Returns:
        The responsible department name
    """
    return CATEGORY_TO_DEPARTMENT.get(
        category,
        DepartmentEnum.ADMINISTRATIVE.value
    )


def determine_urgency(text: str, category: str) -> str:
    """
    Determine the urgency level based on text content and category.

    Args:
        text: The complaint text (lowercase)
        category: The predicted category

    Returns:
        Urgency level (High, Medium, or Low)
    """
    text_lower = text.lower()

    # Check for high urgency keywords
    for keyword in HIGH_URGENCY_KEYWORDS:
        if keyword in text_lower:
            return UrgencyEnum.HIGH.value

    # Check for medium urgency keywords
    for keyword in MEDIUM_URGENCY_KEYWORDS:
        if keyword in text_lower:
            return UrgencyEnum.MEDIUM.value

    # Fall back to category-based default urgency
    return CATEGORY_DEFAULT_URGENCY.get(
        category,
        UrgencyEnum.LOW.value
    )


# List of all valid categories for the model
VALID_CATEGORIES: List[str] = [
    CategoryEnum.CORRUPTION.value,
    CategoryEnum.UTILITY_ISSUE.value,
    CategoryEnum.SERVICE_DELAY.value,
    CategoryEnum.HARASSMENT.value,
    CategoryEnum.FINANCIAL_ISSUE.value,
    CategoryEnum.LAW_ENFORCEMENT.value,
]

