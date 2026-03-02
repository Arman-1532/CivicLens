"""
Text preprocessing service for NLP operations.
Handles text cleaning, normalization, and preparation for ML model.
"""

import re
import string
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Common English stopwords (avoiding NLTK dependency for basic operation)
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


class TextPreprocessor:
    """
    Text preprocessing class for complaint classification.
    """

    def __init__(self, remove_stopwords: bool = True, min_word_length: int = 2):
        """
        Initialize the preprocessor.

        Args:
            remove_stopwords: Whether to remove stopwords
            min_word_length: Minimum word length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.stopwords = STOPWORDS

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for classification.

        Args:
            text: Raw input text

        Returns:
            Cleaned and preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""

        try:
            # Convert to lowercase
            text = text.lower()

            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)

            # Remove phone numbers
            text = re.sub(r'\b\d{10,}\b', '', text)
            text = re.sub(r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}', '', text)

            # Remove special characters and digits (keep only letters and spaces)
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)

            # Tokenize
            words = text.split()

            # Remove stopwords and short words
            if self.remove_stopwords:
                words = [
                    word for word in words
                    if word not in self.stopwords
                    and len(word) >= self.min_word_length
                ]
            else:
                words = [
                    word for word in words
                    if len(word) >= self.min_word_length
                ]

            # Join back to string
            cleaned_text = ' '.join(words)

            return cleaned_text.strip()

        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text.lower().strip() if text else ""

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of raw texts

        Returns:
            List of preprocessed texts
        """
        return [self.clean_text(text) for text in texts]


# Create default preprocessor instance
default_preprocessor = TextPreprocessor()


def preprocess_text(text: str) -> str:
    """
    Convenience function to preprocess a single text.

    Args:
        text: Raw input text

    Returns:
        Preprocessed text
    """
    return default_preprocessor.clean_text(text)


def preprocess_batch(texts: List[str]) -> List[str]:
    """
    Convenience function to preprocess multiple texts.

    Args:
        texts: List of raw texts

    Returns:
        List of preprocessed texts
    """
    return default_preprocessor.preprocess_batch(texts)

