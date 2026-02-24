"""
Cross-validation module for complaint classification.
Provides k-fold cross-validation and hyperparameter tuning.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing.merge_datasets import create_training_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_pipeline(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    svm_kernel: str = 'linear',
    svm_c: float = 1.0
) -> Pipeline:
    """
    Create a scikit-learn pipeline for text classification.

    Args:
        max_features: Maximum TF-IDF features
        ngram_range: N-gram range
        svm_kernel: SVM kernel type
        svm_c: SVM C parameter

    Returns:
        sklearn Pipeline
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )),
        ('classifier', SVC(
            kernel=svm_kernel,
            C=svm_c,
            probability=True,
            class_weight='balanced',
            random_state=42
        ))
    ])


def cross_validate(
    X: pd.Series,
    y: pd.Series,
    n_folds: int = 5,
    scoring: str = 'f1_weighted'
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.

    Args:
        X: Text data
        y: Labels
        n_folds: Number of folds
        scoring: Scoring metric

    Returns:
        Dictionary with CV results
    """
    logger.info(f"Starting {n_folds}-fold cross-validation...")

    # Create pipeline
    pipeline = create_pipeline()

    # Create stratified k-fold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    results = {
        'n_folds': n_folds,
        'scoring': scoring,
        'scores': scores.tolist(),
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'min_score': scores.min(),
        'max_score': scores.max()
    }

    logger.info(f"CV Results - Mean: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")

    return results


def grid_search_cv(
    X: pd.Series,
    y: pd.Series,
    param_grid: Optional[Dict[str, List]] = None,
    n_folds: int = 3,
    scoring: str = 'f1_weighted'
) -> Dict[str, Any]:
    """
    Perform grid search with cross-validation for hyperparameter tuning.

    Args:
        X: Text data
        y: Labels
        param_grid: Parameter grid for search
        n_folds: Number of CV folds
        scoring: Scoring metric

    Returns:
        Dictionary with best parameters and results
    """
    logger.info("Starting grid search cross-validation...")

    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf']
        }

    # Create pipeline
    pipeline = create_pipeline()

    # Create stratified k-fold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    logger.info(f"Searching over {len(param_grid)} parameters...")
    grid_search.fit(X, y)

    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': {
            'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
            'mean_train_score': grid_search.cv_results_['mean_train_score'].tolist(),
            'params': [str(p) for p in grid_search.cv_results_['params']]
        }
    }

    logger.info(f"Best Score: {results['best_score']:.4f}")
    logger.info(f"Best Parameters: {results['best_params']}")

    return results


def print_cv_report(results: Dict[str, Any]) -> None:
    """
    Print cross-validation results report.

    Args:
        results: CV results dictionary
    """
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION REPORT")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Folds: {results['n_folds']}")
    print(f"  Scoring: {results['scoring']}")

    print(f"\nResults:")
    print(f"  Mean Score:  {results['mean_score']:.4f}")
    print(f"  Std Dev:     {results['std_score']:.4f}")
    print(f"  Min Score:   {results['min_score']:.4f}")
    print(f"  Max Score:   {results['max_score']:.4f}")

    print(f"\nPer-Fold Scores:")
    for i, score in enumerate(results['scores'], 1):
        print(f"  Fold {i}: {score:.4f}")

    print("\n" + "=" * 60)


def print_grid_search_report(results: Dict[str, Any]) -> None:
    """
    Print grid search results report.

    Args:
        results: Grid search results dictionary
    """
    print("\n" + "=" * 60)
    print("GRID SEARCH REPORT")
    print("=" * 60)

    print(f"\nBest Score: {results['best_score']:.4f}")

    print(f"\nBest Parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")

    print(f"\nTop 5 Parameter Combinations:")
    cv_results = results['cv_results']
    indices = np.argsort(cv_results['mean_test_score'])[::-1][:5]

    for i, idx in enumerate(indices, 1):
        print(f"\n  [{i}] Score: {cv_results['mean_test_score'][idx]:.4f} (+/- {cv_results['std_test_score'][idx]:.4f})")
        print(f"      Params: {cv_results['params'][idx][:80]}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = create_training_data(balance_classes=True)

    X = df['text']
    y = df['category']

    print(f"Data loaded: {len(df)} samples")

    # Run cross-validation
    print("\n" + "=" * 60)
    print("Running Cross-Validation...")
    print("=" * 60)

    cv_results = cross_validate(X, y, n_folds=5)
    print_cv_report(cv_results)

    # Run grid search (optional - can be slow)
    run_grid_search = False  # Set to True to run grid search

    if run_grid_search:
        print("\n" + "=" * 60)
        print("Running Grid Search (this may take a while)...")
        print("=" * 60)

        # Smaller grid for faster search
        small_param_grid = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1, 2)],
            'classifier__C': [0.1, 1.0],
            'classifier__kernel': ['linear']
        }

        gs_results = grid_search_cv(X, y, param_grid=small_param_grid, n_folds=3)
        print_grid_search_report(gs_results)

