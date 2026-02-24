"""
Model evaluation module for complaint classification.
Provides detailed evaluation metrics and visualization.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train_model import ComplaintClassifier, ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(
    model: ComplaintClassifier,
    X_test: pd.Series,
    y_test: pd.Series,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained ComplaintClassifier
        X_test: Test text data
        y_test: True labels
        detailed: Whether to include detailed metrics

    Returns:
        Dictionary with all evaluation metrics
    """
    logger.info("Evaluating model...")

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Basic metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
    }

    if detailed:
        # Per-class metrics
        results['classification_report'] = classification_report(
            y_test, y_pred, output_dict=True
        )
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        results['classes'] = model.classes_.tolist()

        # Confidence analysis
        confidences = np.max(y_proba, axis=1)
        results['avg_confidence'] = np.mean(confidences)
        results['min_confidence'] = np.min(confidences)
        results['max_confidence'] = np.max(confidences)

        # Per-class confidence
        class_confidences = {}
        for cls in model.classes_:
            mask = y_test == cls
            if mask.sum() > 0:
                class_confidences[cls] = np.mean(confidences[mask])
        results['class_confidences'] = class_confidences

        # Error analysis
        errors = y_test != y_pred
        results['error_rate'] = errors.mean()
        results['num_errors'] = errors.sum()

        # Misclassification details
        if errors.sum() > 0:
            error_df = pd.DataFrame({
                'text': X_test[errors].values,
                'true_label': y_test[errors].values,
                'predicted': y_pred[errors],
                'confidence': confidences[errors]
            })
            results['error_samples'] = error_df.head(10).to_dict('records')

    return results


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """
    Print a formatted evaluation report.

    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    print("\n📊 Overall Metrics:")
    print(f"  Accuracy:           {results['accuracy']:.4f}")
    print(f"  Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {results['recall_weighted']:.4f}")
    print(f"  F1 Score (weighted):  {results['f1_weighted']:.4f}")

    print(f"\n  Precision (macro):    {results['precision_macro']:.4f}")
    print(f"  Recall (macro):       {results['recall_macro']:.4f}")
    print(f"  F1 Score (macro):     {results['f1_macro']:.4f}")

    if 'classification_report' in results:
        print("\n📋 Per-Class Metrics:")
        report = results['classification_report']

        # Header
        print(f"  {'Category':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("  " + "-" * 65)

        for cls in results.get('classes', []):
            if cls in report:
                m = report[cls]
                print(f"  {cls:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")

    if 'avg_confidence' in results:
        print("\n🎯 Confidence Analysis:")
        print(f"  Average Confidence: {results['avg_confidence']:.4f}")
        print(f"  Min Confidence:     {results['min_confidence']:.4f}")
        print(f"  Max Confidence:     {results['max_confidence']:.4f}")

        if 'class_confidences' in results:
            print("\n  Per-Class Average Confidence:")
            for cls, conf in results['class_confidences'].items():
                print(f"    {cls}: {conf:.4f}")

    if 'error_rate' in results:
        print("\n❌ Error Analysis:")
        print(f"  Error Rate:    {results['error_rate']:.4f}")
        print(f"  Total Errors:  {results['num_errors']}")

        if 'error_samples' in results and results['error_samples']:
            print("\n  Sample Misclassifications:")
            for i, err in enumerate(results['error_samples'][:5], 1):
                print(f"\n  [{i}] Text: {err['text'][:50]}...")
                print(f"      True: {err['true_label']} | Predicted: {err['predicted']} | Conf: {err['confidence']:.2f}")

    if 'confusion_matrix' in results:
        print("\n📊 Confusion Matrix:")
        cm = results['confusion_matrix']
        classes = results.get('classes', [f'Class {i}' for i in range(len(cm))])

        # Print header
        print(f"  {'':>20}", end="")
        for cls in classes:
            print(f"{cls[:8]:>10}", end="")
        print()

        # Print rows
        for i, row in enumerate(cm):
            print(f"  {classes[i][:20]:>20}", end="")
            for val in row:
                print(f"{val:>10}", end="")
            print()

    print("\n" + "=" * 60)


def compare_models(
    models: List[Dict[str, Any]],
    X_test: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compare multiple models.

    Args:
        models: List of dicts with 'name' and 'model' keys
        X_test: Test text data
        y_test: True labels

    Returns:
        DataFrame with comparison results
    """
    results = []

    for model_info in models:
        name = model_info['name']
        model = model_info['model']

        logger.info(f"Evaluating {name}...")
        eval_results = evaluate_model(model, X_test, y_test, detailed=False)

        results.append({
            'Model': name,
            'Accuracy': eval_results['accuracy'],
            'Precision': eval_results['precision_weighted'],
            'Recall': eval_results['recall_weighted'],
            'F1 Score': eval_results['f1_weighted']
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    from train_model import train_model
    from sklearn.model_selection import train_test_split
    from data_processing.merge_datasets import create_training_data

    # Load or create data
    print("Loading data...")
    df = create_training_data(balance_classes=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['category'],
        test_size=0.2,
        random_state=42,
        stratify=df['category']
    )

    # Train model
    print("Training model...")
    model = ComplaintClassifier()
    model.fit(X_train, y_train)

    # Evaluate
    results = evaluate_model(model, X_test, y_test, detailed=True)
    print_evaluation_report(results)

