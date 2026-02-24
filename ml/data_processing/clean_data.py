"""
Data cleaning module for complaint classification.
Handles text preprocessing and data quality improvements.
"""

import re
import logging
from typing import List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import nltk
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Common English stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
}


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True, min_word_length: int = 2) -> str:
    """
    Clean and preprocess a single text string.

    Args:
        text: Raw input text
        remove_stopwords: Whether to remove stopwords
        lemmatize: Whether to lemmatize words
        min_word_length: Minimum word length to keep

    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{10,}\b', '', text)
    text = re.sub(r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Tokenize and filter
    words = text.split()

    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    
    if lemmatize:
        words = [lemmatizer.lemmatize(w) for w in words]

    words = [w for w in words if len(w) >= min_word_length]

    return ' '.join(words).strip()


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'category',
    remove_stopwords: bool = True,
    min_text_length: int = 10
) -> pd.DataFrame:
    """
    Clean a DataFrame containing complaint data.

    Args:
        df: Input DataFrame
        text_column: Name of the text column
        label_column: Name of the label column
        remove_stopwords: Whether to remove stopwords
        min_text_length: Minimum text length after cleaning

    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning dataframe with {len(df)} rows")

    # Create a copy
    df_clean = df.copy()

    # Remove rows with missing text or labels
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=[text_column, label_column])
    logger.info(f"Removed {initial_count - len(df_clean)} rows with missing values")

    # Clean text
    df_clean['cleaned_text'] = df_clean[text_column].apply(
        lambda x: clean_text(x, remove_stopwords=remove_stopwords)
    )

    # Remove rows with text too short after cleaning
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['cleaned_text'].str.len() >= min_text_length]
    logger.info(f"Removed {initial_count - len(df_clean)} rows with text too short")

    # Remove duplicates
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['cleaned_text'])
    logger.info(f"Removed {initial_count - len(df_clean)} duplicate rows")

    logger.info(f"Final cleaned dataset: {len(df_clean)} rows")

    return df_clean


def remove_outliers(df: pd.DataFrame, text_column: str = 'cleaned_text',
                    min_words: int = 3, max_words: int = 500) -> pd.DataFrame:
    """
    Remove outliers based on text length.

    Args:
        df: Input DataFrame
        text_column: Name of the text column
        min_words: Minimum number of words
        max_words: Maximum number of words

    Returns:
        DataFrame with outliers removed
    """
    df_filtered = df.copy()
    df_filtered['word_count'] = df_filtered[text_column].apply(lambda x: len(x.split()))

    initial_count = len(df_filtered)
    df_filtered = df_filtered[(df_filtered['word_count'] >= min_words) &
                              (df_filtered['word_count'] <= max_words)]

    logger.info(f"Removed {initial_count - len(df_filtered)} outlier rows based on word count")

    df_filtered = df_filtered.drop(columns=['word_count'])

    return df_filtered


if __name__ == "__main__":
    # Test the cleaning functions
    test_texts = [
        "The water supply has been IRREGULAR for 3 days!!! Call 1234567890",
        "Email support@example.com for help with https://website.com issue",
        "URGENT: Police not responding to emergency calls in our area",
    ]

    print("=== Text Cleaning Test ===")
    for text in test_texts:
        cleaned = clean_text(text)
        print(f"Original: {text[:50]}...")
        print(f"Cleaned:  {cleaned}")
        print()

