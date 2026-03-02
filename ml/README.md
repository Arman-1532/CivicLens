# CivicLens ML Training Pipeline

Machine Learning pipeline for training the complaint classification model.

## Overview

This module provides:
- **Data Processing**: Clean, merge, and prepare complaint datasets
- **Label Mapping**: Map various dataset labels to unified governance categories
- **Model Training**: Train TF-IDF + SVM classifier
- **Evaluation**: Comprehensive model evaluation with metrics
- **Cross-Validation**: K-fold CV and hyperparameter tuning

## Project Structure

```
ml/
├── data_processing/
│   ├── __init__.py
│   ├── clean_data.py       # Text cleaning and preprocessing
│   ├── label_mapping.py    # Map labels to unified categories
│   └── merge_datasets.py   # Load and merge multiple datasets
│
├── training/
│   ├── __init__.py
│   ├── train_model.py      # Main training script
│   ├── evaluate_model.py   # Model evaluation metrics
│   └── cross_validation.py # K-fold CV and grid search
│
├── artifacts/
│   ├── classifier.pkl      # Trained SVM model
│   └── tfidf_vectorizer.pkl # Fitted TF-IDF vectorizer
│
└── README.md
```

## Categories

The model classifies complaints into 6 unified categories:

| Category | Description |
|----------|-------------|
| Corruption | Bribery, fraud, misuse of funds |
| Utility Issue | Water, electricity, roads, sanitation |
| Service Delay | Pending applications, bureaucratic delays |
| Harassment | Workplace harassment, discrimination |
| Financial Issue | Billing, taxes, refunds, payments |
| Law Enforcement Issue | Police, crime, safety concerns |

## Installation

```bash
# From the project root
cd ml

# Install dependencies (if not already installed)
pip install pandas numpy scikit-learn joblib
```

## Usage

### 1. Train the Model

```bash
cd /path/to/CivicLens/ml
python -m training.train_model
```

This will:
1. Create/load training data
2. Clean and preprocess text
3. Map labels to unified categories
4. Train TF-IDF vectorizer
5. Train SVM classifier
6. Evaluate on test set
7. Save models to `artifacts/` and `backend/trained_models/`

### 2. Evaluate the Model

```bash
python -m training.evaluate_model
```

### 3. Run Cross-Validation

```bash
python -m training.cross_validation
```

### 4. Process Data Only

```bash
python -m data_processing.merge_datasets
```

## Training Output

After training, you'll see output like:

```
============================================================
Starting Model Training Pipeline
============================================================
Creating training data...
Training data: 1200 samples
Categories: 6

Training TF-IDF vectorizer...
Vocabulary size: 2456
TF-IDF matrix shape: (960, 2456)

Training SVM classifier...
Training complete!

============================================================
Model Evaluation
============================================================

Accuracy: 0.9542

Classification Report:
  Corruption:
    Precision: 0.9500
    Recall: 0.9500
    F1-Score: 0.9500
  Utility Issue:
    Precision: 0.9750
    Recall: 0.9750
    F1-Score: 0.9750
  ...

Weighted Avg F1: 0.9542

============================================================
Saving Model Artifacts
============================================================
Saved classifier to ml/artifacts/classifier.pkl
Saved vectorizer to ml/artifacts/tfidf_vectorizer.pkl
Models copied to backend: backend/trained_models

============================================================
Training Pipeline Complete!
============================================================
```

## Using Custom Datasets

Place your CSV files in `data/raw/` with one of these configurations:

1. **kaggle_customer_complaints.csv**
   - `complaint`: Text column
   - `category`: Label column

2. **kaggle_public_service.csv**
   - `description`: Text column
   - `type`: Label column

3. **kaggle_consumer_complaints.csv**
   - `consumer_complaint_narrative`: Text column
   - `product`: Label column

The pipeline will automatically detect and use available datasets.

## API

### Training

```python
from ml.training import train_model, ComplaintClassifier

# Train and save model
model, results = train_model(save_to_backend=True)

# Or load existing model
model = ComplaintClassifier.load(
    Path("artifacts/classifier.pkl"),
    Path("artifacts/tfidf_vectorizer.pkl")
)

# Make predictions
predictions = model.predict(pd.Series(["Water supply issue in my area"]))
probabilities = model.predict_proba(pd.Series(["Water supply issue"]))
```

### Data Processing

```python
from ml.data_processing import clean_text, create_training_data

# Clean single text
cleaned = clean_text("The water supply has been IRREGULAR!!!")
# Result: "water supply irregular"

# Create full training dataset
df = create_training_data(balance_classes=True)
```

### Evaluation

```python
from ml.training import evaluate_model, print_evaluation_report

results = evaluate_model(model, X_test, y_test, detailed=True)
print_evaluation_report(results)
```

## Model Architecture

- **Vectorizer**: TF-IDF with:
  - Max 10,000 features
  - Unigrams and bigrams (1,2)
  - English stopwords removed
  - Min document frequency: 2
  - Max document frequency: 95%

- **Classifier**: Support Vector Machine (SVM) with:
  - Linear kernel
  - C=1.0
  - Probability estimates enabled
  - Balanced class weights

## Files Generated

After training:

```
ml/artifacts/
├── classifier.pkl          # ~500KB-2MB
└── tfidf_vectorizer.pkl    # ~1-5MB

backend/trained_models/
├── classifier.pkl          # Copy for backend
└── tfidf_vectorizer.pkl    # Copy for backend

data/
├── interim/
│   ├── merged_dataset.csv
│   └── cleaned_dataset.csv
└── processed/
    └── final_training_data.csv
```

## License

MIT License

