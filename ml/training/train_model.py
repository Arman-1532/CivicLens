"""
Model training module for complaint classification.
Trains TF-IDF vectorizer and SVM classifier.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing.merge_datasets import create_training_data, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ML_DIR = BASE_DIR / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"
BACKEND_MODELS_DIR = BASE_DIR / "backend" / "trained_models"


class ComplaintClassifier:
    """
    Complaint classification model using TF-IDF + SVM.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        n_estimators: int = 200,
        max_depth: Optional[int] = 20,
        random_state: int = 42
    ):
        """
        Initialize the classifier.

        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams for TF-IDF
            svm_kernel: SVM kernel type
            svm_c: SVM regularization parameter
            random_state: Random seed for reproducibility
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.9
        )

        # Use CatBoost for better performance and easier installation on Mac
        self.classifier = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function='MultiClass',
            random_seed=random_state,
            verbose=False,
            thread_count=-1
        )
        
        self.label_encoder = LabelEncoder()

        self.classes_ = None
        self.is_trained = False

    def fit(self, X: pd.Series, y: pd.Series) -> 'ComplaintClassifier':
        """
        Train the model on the given data.

        Args:
            X: Text data (complaints)
            y: Labels (categories)

        Returns:
            self
        """
        logger.info("Training TF-IDF vectorizer...")
        X_tfidf = self.vectorizer.fit_transform(X)
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"TF-IDF matrix shape: {X_tfidf.shape}")

        logger.info("Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_

        logger.info("Training CatBoost classifier...")
        self.classifier.fit(X_tfidf.toarray(), y_encoded)
        self.is_trained = True

        logger.info("Training complete!")
        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        """
        Predict categories for given texts.

        Args:
            X: Text data to classify

        Returns:
            Predicted categories
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_tfidf = self.vectorizer.transform(X)
        predictions_encoded = self.classifier.predict(X_tfidf)
        return self.label_encoder.inverse_transform(predictions_encoded)

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """
        Predict category probabilities for given texts.

        Args:
            X: Text data to classify

        Returns:
            Probability matrix
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_tfidf.toarray())

    def evaluate(self, X: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            X: Test text data
            y: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)

        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y, predictions)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions
        }

    def save(self, model_path: Path, vectorizer_path: Path) -> None:
        """
        Save the trained model and vectorizer to disk.

        Args:
            model_path: Path to save the classifier
            vectorizer_path: Path to save the vectorizer
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Ensure directories exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)

        # Save artifacts
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, str(model_path).replace('classifier.pkl', 'label_encoder.pkl'))

        logger.info(f"Saved classifier to {model_path}")
        logger.info(f"Saved vectorizer to {vectorizer_path}")

    @classmethod
    def load(cls, model_path: Path, vectorizer_path: Path) -> 'ComplaintClassifier':
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the classifier file
            vectorizer_path: Path to the vectorizer file

        Returns:
            Loaded ComplaintClassifier instance
        """
        instance = cls()
        instance.classifier = joblib.load(model_path)
        instance.vectorizer = joblib.load(vectorizer_path)
        instance.label_encoder = joblib.load(str(model_path).replace('classifier.pkl', 'label_encoder.pkl'))
        instance.classes_ = instance.label_encoder.classes_
        instance.is_trained = True

        logger.info(f"Loaded classifier from {model_path}")
        logger.info(f"Loaded vectorizer from {vectorizer_path}")

        return instance


def train_model(
    data_path: Optional[Path] = None,
    test_size: float = 0.2,
    save_to_backend: bool = True
) -> Tuple[ComplaintClassifier, Dict[str, Any]]:
    """
    Main training function.

    Args:
        data_path: Path to training data CSV (optional, will generate if not provided)
        test_size: Fraction of data to use for testing
        save_to_backend: Whether to copy models to backend directory

    Returns:
        Tuple of (trained model, evaluation results)
    """
    logger.info("=" * 60)
    logger.info("Starting Model Training Pipeline")
    logger.info("=" * 60)

    # Load or create training data
    if data_path and data_path.exists():
        logger.info(f"Loading training data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        logger.info("Creating training data...")
        df = create_training_data(balance_classes=True)

    logger.info(f"Training data: {len(df)} samples")
    logger.info(f"Categories: {df['category'].nunique()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['category'],
        test_size=test_size,
        random_state=42,
        stratify=df['category']
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Initialize and train model
    model = ComplaintClassifier(
        max_features=10000,
        ngram_range=(1, 2),
        n_estimators=300,
        max_depth=None
    )

    model.fit(X_train, y_train)

    # Evaluate model
    logger.info("\n" + "=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)

    results = model.evaluate(X_test, y_test)

    logger.info(f"\nAccuracy: {results['accuracy']:.4f}")
    logger.info("\nClassification Report:")

    # Print classification report
    report = results['classification_report']
    for category in model.classes_:
        if category in report:
            metrics = report[category]
            logger.info(f"  {category}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
            logger.info(f"    F1-Score: {metrics['f1-score']:.4f}")

    logger.info(f"\nWeighted Avg F1: {report['weighted avg']['f1-score']:.4f}")

    # Save model artifacts
    logger.info("\n" + "=" * 60)
    logger.info("Saving Model Artifacts")
    logger.info("=" * 60)

    # Save to ML artifacts directory
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(
        ARTIFACTS_DIR / "classifier.pkl",
        ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
    )

    # Copy to backend if requested
    if save_to_backend:
        BACKEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(
            BACKEND_MODELS_DIR / "classifier.pkl",
            BACKEND_MODELS_DIR / "tfidf_vectorizer.pkl"
        )
        logger.info(f"Models copied to backend: {BACKEND_MODELS_DIR}")

    logger.info("\n" + "=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info("=" * 60)

    return model, results


if __name__ == "__main__":
    # Run training
    model, results = train_model(save_to_backend=True)

    # Test with sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)

    test_complaints = [
        "The water supply has been irregular for the past week in our area",
        "Government official asked for bribe to process my application",
        "Police are not taking action on my theft complaint",
        "My electricity bill shows incorrect charges this month",
        "Passport application pending for six months with no response",
        "Facing harassment from supervisor at workplace",
    ]

    for complaint in test_complaints:
        prediction = model.predict(pd.Series([complaint]))[0]
        probas = model.predict_proba(pd.Series([complaint]))[0]
        confidence = max(probas)
        print(f"\nComplaint: {complaint[:50]}...")
        print(f"Category: {prediction}")
        print(f"Confidence: {confidence:.2%}")

