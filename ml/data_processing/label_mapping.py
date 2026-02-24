"""
Label mapping module for complaint classification.
Maps original dataset labels to unified governance categories.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Unified governance categories
UNIFIED_CATEGORIES = [
    "Corruption",
    "Utility Issue",
    "Service Delay",
    "Harassment",
    "Financial Issue",
    "Law Enforcement Issue"
]

# Mapping from various dataset labels to unified categories
# This handles different naming conventions across datasets
LABEL_MAPPING: Dict[str, str] = {
    # Corruption related
    "corruption": "Corruption",
    "bribery": "Corruption",
    "fraud": "Corruption",
    "scam": "Corruption",
    "embezzlement": "Corruption",
    "misuse of funds": "Corruption",
    "kickback": "Corruption",

    # Utility issues
    "utility issue": "Utility Issue",
    "utility": "Utility Issue",
    "water": "Utility Issue",
    "water supply": "Utility Issue",
    "electricity": "Utility Issue",
    "power": "Utility Issue",
    "power outage": "Utility Issue",
    "gas": "Utility Issue",
    "sewage": "Utility Issue",
    "drainage": "Utility Issue",
    "sanitation": "Utility Issue",
    "garbage": "Utility Issue",
    "waste": "Utility Issue",
    "infrastructure": "Utility Issue",
    "road": "Utility Issue",
    "roads": "Utility Issue",
    "streetlight": "Utility Issue",
    "public utilities": "Utility Issue",

    # Service delays
    "service delay": "Service Delay",
    "delay": "Service Delay",
    "delayed": "Service Delay",
    "pending": "Service Delay",
    "slow service": "Service Delay",
    "poor service": "Service Delay",
    "bad service": "Service Delay",
    "service issue": "Service Delay",
    "customer service": "Service Delay",
    "response time": "Service Delay",
    "waiting": "Service Delay",
    "bureaucracy": "Service Delay",
    "red tape": "Service Delay",
    "administrative": "Service Delay",

    # Harassment
    "harassment": "Harassment",
    "abuse": "Harassment",
    "discrimination": "Harassment",
    "sexual harassment": "Harassment",
    "workplace harassment": "Harassment",
    "bullying": "Harassment",
    "intimidation": "Harassment",
    "threat": "Harassment",
    "threats": "Harassment",
    "hostile": "Harassment",

    # Financial issues
    "financial issue": "Financial Issue",
    "financial": "Financial Issue",
    "billing": "Financial Issue",
    "billing issue": "Financial Issue",
    "overcharge": "Financial Issue",
    "overcharging": "Financial Issue",
    "refund": "Financial Issue",
    "payment": "Financial Issue",
    "fee": "Financial Issue",
    "fees": "Financial Issue",
    "tax": "Financial Issue",
    "taxes": "Financial Issue",
    "pension": "Financial Issue",
    "salary": "Financial Issue",
    "compensation": "Financial Issue",
    "money": "Financial Issue",
    "debt": "Financial Issue",
    "loan": "Financial Issue",
    "bank": "Financial Issue",
    "banking": "Financial Issue",
    "credit": "Financial Issue",
    "credit card": "Financial Issue",
    "mortgage": "Financial Issue",

    # Law enforcement
    "law enforcement issue": "Law Enforcement Issue",
    "law enforcement": "Law Enforcement Issue",
    "police": "Law Enforcement Issue",
    "crime": "Law Enforcement Issue",
    "criminal": "Law Enforcement Issue",
    "theft": "Law Enforcement Issue",
    "robbery": "Law Enforcement Issue",
    "violence": "Law Enforcement Issue",
    "assault": "Law Enforcement Issue",
    "safety": "Law Enforcement Issue",
    "security": "Law Enforcement Issue",
    "emergency": "Law Enforcement Issue",
    "accident": "Law Enforcement Issue",
    "traffic": "Law Enforcement Issue",
    "illegal": "Law Enforcement Issue",
}

# Keywords for classification when label mapping fails
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Corruption": [
        "bribe", "corrupt", "fraud", "scam", "embezzle", "kickback",
        "misuse", "nepotism", "favoritism", "illegal payment"
    ],
    "Utility Issue": [
        "water", "electricity", "power", "gas", "sewage", "drainage",
        "garbage", "waste", "road", "streetlight", "infrastructure",
        "supply", "outage", "sanitation", "utility"
    ],
    "Service Delay": [
        "delay", "slow", "waiting", "pending", "late", "overdue",
        "not responding", "no response", "poor service", "bad service",
        "bureaucracy", "red tape", "inefficient"
    ],
    "Harassment": [
        "harass", "abuse", "discriminat", "bully", "intimidat", "threat",
        "hostile", "mistreat", "victim", "assault", "sexual"
    ],
    "Financial Issue": [
        "bill", "payment", "refund", "overcharge", "fee", "tax",
        "pension", "salary", "money", "bank", "loan", "credit",
        "debt", "mortgage", "financial"
    ],
    "Law Enforcement Issue": [
        "police", "crime", "theft", "robbery", "violence", "assault",
        "safety", "security", "emergency", "accident", "traffic",
        "illegal", "law enforcement", "criminal"
    ]
}


def map_label(label: str) -> Optional[str]:
    """
    Map a single label to unified category.

    Args:
        label: Original label from dataset

    Returns:
        Unified category or None if no mapping found
    """
    if not label or not isinstance(label, str):
        return None

    # Normalize the label
    label_lower = label.lower().strip()

    # Direct mapping
    if label_lower in LABEL_MAPPING:
        return LABEL_MAPPING[label_lower]

    # Check if label contains any mapped key
    for key, category in LABEL_MAPPING.items():
        if key in label_lower or label_lower in key:
            return category

    # If label is already a unified category
    for unified in UNIFIED_CATEGORIES:
        if unified.lower() == label_lower:
            return unified

    return None


def classify_by_keywords(text: str) -> Optional[str]:
    """
    Classify text based on keywords when label mapping fails.

    Args:
        text: The complaint text

    Returns:
        Predicted category based on keywords or None
    """
    if not text or not isinstance(text, str):
        return None

    text_lower = text.lower()

    # Count keyword matches for each category
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score

    if scores:
        # Return category with highest score
        return max(scores, key=scores.get)

    return None


def map_labels_in_dataframe(
    df: pd.DataFrame,
    label_column: str = 'category',
    text_column: str = 'text',
    use_keyword_fallback: bool = True
) -> pd.DataFrame:
    """
    Map all labels in a DataFrame to unified categories.

    Args:
        df: Input DataFrame
        label_column: Name of the label column
        text_column: Name of the text column (for keyword fallback)
        use_keyword_fallback: Whether to use keyword classification as fallback

    Returns:
        DataFrame with mapped labels
    """
    logger.info(f"Mapping labels for {len(df)} rows")

    df_mapped = df.copy()

    # Map labels
    df_mapped['mapped_category'] = df_mapped[label_column].apply(map_label)

    # Count unmapped
    unmapped_count = df_mapped['mapped_category'].isna().sum()
    logger.info(f"Direct mapping: {len(df) - unmapped_count} mapped, {unmapped_count} unmapped")

    # Use keyword fallback for unmapped labels
    if use_keyword_fallback and unmapped_count > 0:
        unmapped_mask = df_mapped['mapped_category'].isna()
        df_mapped.loc[unmapped_mask, 'mapped_category'] = df_mapped.loc[
            unmapped_mask, text_column
        ].apply(classify_by_keywords)

        still_unmapped = df_mapped['mapped_category'].isna().sum()
        logger.info(f"After keyword fallback: {unmapped_count - still_unmapped} additional mapped")

    # Remove rows that couldn't be mapped
    initial_count = len(df_mapped)
    df_mapped = df_mapped.dropna(subset=['mapped_category'])
    logger.info(f"Removed {initial_count - len(df_mapped)} rows with unmappable labels")

    # Log category distribution
    logger.info("Category distribution:")
    for cat, count in df_mapped['mapped_category'].value_counts().items():
        logger.info(f"  {cat}: {count}")

    return df_mapped


def get_unified_categories() -> List[str]:
    """Get list of unified categories."""
    return UNIFIED_CATEGORIES.copy()


if __name__ == "__main__":
    # Test the mapping functions
    test_labels = [
        "water supply",
        "Police Complaint",
        "billing issue",
        "CORRUPTION",
        "harassment at workplace",
        "unknown_label"
    ]

    print("=== Label Mapping Test ===")
    for label in test_labels:
        mapped = map_label(label)
        print(f"'{label}' -> '{mapped}'")

    print("\n=== Keyword Classification Test ===")
    test_texts = [
        "The water supply has been irregular for days",
        "I was overcharged on my electricity bill",
        "Police are not responding to our calls",
    ]

    for text in test_texts:
        category = classify_by_keywords(text)
        print(f"'{text[:40]}...' -> '{category}'")

