"""
Prediction service for complaint classification.
Handles model loading and inference.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import joblib
import numpy as np

from ..core.config import settings
from ..models.complaint_model import (
    get_department_for_category,
    determine_urgency,
    VALID_CATEGORIES
)
from .preprocessing import preprocess_text

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service class for loading ML models and making predictions.
    """

    def __init__(self):
        """Initialize the prediction service."""
        self.model = None
        self.vectorizer = None
        self._model_loaded = False
        self._vectorizer_loaded = False

    def load_models(self) -> bool:
        """
        Load the trained model and vectorizer from disk.

        Returns:
            True if both model and vectorizer loaded successfully
        """
        try:
            model_path = settings.model_full_path
            vectorizer_path = settings.vectorizer_full_path

            # Load classifier
            if model_path.exists():
                self.model = joblib.load(model_path)
                self._model_loaded = True
                logger.info(f"Model loaded successfully from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}")
                self._model_loaded = False

            # Load vectorizer
            if vectorizer_path.exists():
                self.vectorizer = joblib.load(vectorizer_path)
                self._vectorizer_loaded = True
                logger.info(f"Vectorizer loaded successfully from {vectorizer_path}")
            else:
                logger.warning(f"Vectorizer file not found at {vectorizer_path}")
                self._vectorizer_loaded = False

            return self._model_loaded and self._vectorizer_loaded

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._model_loaded = False
            self._vectorizer_loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        """Check if the service is ready to make predictions."""
        return self._model_loaded and self._vectorizer_loaded

    @property
    def model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    @property
    def vectorizer_loaded(self) -> bool:
        """Check if vectorizer is loaded."""
        return self._vectorizer_loaded

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make a prediction for the given complaint text.

        Args:
            text: The complaint text to classify

        Returns:
            Dictionary containing prediction results

        Raises:
            RuntimeError: If models are not loaded
            ValueError: If text is empty after preprocessing
        """
        if not self.is_ready:
            raise RuntimeError("Models not loaded. Please ensure model files exist.")

        # Preprocess the text
        cleaned_text = preprocess_text(text)

        if not cleaned_text:
            raise ValueError("Text is empty after preprocessing")

        try:
            # Vectorize the text
            text_vectorized = self.vectorizer.transform([cleaned_text])

            # Get prediction and probability
            prediction = self.model.predict(text_vectorized)[0]

            # Get confidence score
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(text_vectorized)[0]
                confidence = float(np.max(probabilities))
            elif hasattr(self.model, 'decision_function'):
                # For SVM without probability=True
                decision_scores = self.model.decision_function(text_vectorized)[0]
                # Convert decision function to pseudo-probability using softmax
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                confidence = float(np.max(exp_scores / exp_scores.sum()))
            else:
                confidence = 0.0

            # Get department and urgency
            category = str(prediction)
            department = get_department_for_category(category)
            urgency = determine_urgency(text, category)

            return {
                "category": category,
                "department": department,
                "urgency": urgency,
                "confidence": round(confidence, 4)
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_loaded": self._model_loaded,
            "vectorizer_loaded": self._vectorizer_loaded,
            "ready": self.is_ready,
            "categories": VALID_CATEGORIES
        }

        if self._model_loaded and self.model:
            info["model_type"] = type(self.model).__name__

        if self._vectorizer_loaded and self.vectorizer:
            info["vectorizer_type"] = type(self.vectorizer).__name__
            if hasattr(self.vectorizer, 'vocabulary_'):
                info["vocabulary_size"] = len(self.vectorizer.vocabulary_)

        return info


# Create singleton instance
prediction_service = PredictionService()


def get_prediction_service() -> PredictionService:
    """
    Get the prediction service instance.

    Returns:
        The singleton PredictionService instance
    """
    return prediction_service

