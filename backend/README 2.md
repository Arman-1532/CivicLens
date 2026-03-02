# CivicLens Backend

AI-based Complaint Classification System for E-Governance - Backend API

## Tech Stack

- **Framework**: FastAPI
- **ML**: scikit-learn (SVM + TF-IDF)
- **Validation**: Pydantic
- **Server**: Uvicorn

## Project Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── api/
│   │   └── routes/
│   │       ├── complaint.py    # Complaint classification endpoints
│   │       └── health.py       # Health check endpoints
│   ├── core/
│   │   ├── config.py           # Application configuration
│   │   └── security.py         # Security utilities
│   ├── models/
│   │   ├── schema.py           # Pydantic schemas
│   │   └── complaint_model.py  # Category/Department mappings
│   ├── services/
│   │   ├── prediction_service.py  # ML prediction logic
│   │   └── preprocessing.py       # Text preprocessing
│   ├── db/
│   │   ├── database.py         # Database connection (placeholder)
│   │   └── models.py           # ORM models (placeholder)
│   └── utils/
│       └── __init__.py         # Utility functions
├── trained_models/
│   ├── classifier.pkl          # Trained SVM model
│   └── tfidf_vectorizer.pkl    # TF-IDF vectorizer
├── requirements.txt
├── .env
└── run.py
```

## Setup & Installation

### 1. Create Virtual Environment

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

The `.env` file is pre-configured. Modify as needed:

```env
APP_NAME=CivicLens
APP_VERSION=1.0.0
DEBUG=True
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
MODEL_PATH=trained_models/classifier.pkl
VECTORIZER_PATH=trained_models/tfidf_vectorizer.pkl
LOG_LEVEL=INFO
```

### 4. Train the Model (Required First)

Before running the backend, train the ML model using the training pipeline:

```bash
cd ../ml
python training/train_model.py
```

This will generate `classifier.pkl` and `tfidf_vectorizer.pkl` in `backend/trained_models/`.

### 5. Run the Server

```bash
python run.py
# OR
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Base URL: `http://localhost:8000`

### Root
- `GET /` - API information

### Health
- `GET /api/v1/health` - Health check with model status
- `GET /api/v1/health/ready` - Readiness probe
- `GET /api/v1/health/live` - Liveness probe

### Complaints
- `POST /api/v1/complaints/predict` - Classify a complaint
- `GET /api/v1/complaints/categories` - Get all categories
- `GET /api/v1/complaints/model-info` - Get model information

### Shortcut
- `POST /predict` - Quick prediction endpoint

## API Usage Examples

### Classify a Complaint

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/complaints/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The water supply has been irregular for the past week in our area. We have complained multiple times but no action has been taken."}'
```

**Response:**
```json
{
  "category": "Utility Issue",
  "department": "Public Utilities Department",
  "urgency": "Medium",
  "confidence": 0.87,
  "timestamp": "2026-02-19T10:30:00.000Z"
}
```

### Health Check

**Request:**
```bash
curl "http://localhost:8000/api/v1/health"
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "vectorizer_loaded": true,
  "timestamp": "2026-02-19T10:30:00.000Z"
}
```

### Get Categories

**Request:**
```bash
curl "http://localhost:8000/api/v1/complaints/categories"
```

**Response:**
```json
{
  "categories": [
    {"name": "Corruption", "department": "Anti-Corruption Bureau"},
    {"name": "Utility Issue", "department": "Public Utilities Department"},
    {"name": "Service Delay", "department": "Administrative Services"},
    {"name": "Harassment", "department": "Women's Commission / HR"},
    {"name": "Financial Issue", "department": "Finance Department"},
    {"name": "Law Enforcement Issue", "department": "Police Department / Law Enforcement"}
  ]
}
```

## Complaint Categories

| Category | Department | Default Urgency |
|----------|------------|-----------------|
| Corruption | Anti-Corruption Bureau | High |
| Utility Issue | Public Utilities Department | Medium |
| Service Delay | Administrative Services | Low |
| Harassment | Women's Commission / HR | High |
| Financial Issue | Finance Department | Medium |
| Law Enforcement Issue | Police Department / Law Enforcement | High |

## Swagger Documentation

Access the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Error Handling

The API returns structured error responses:

```json
{
  "error": "Validation Error",
  "message": "Request validation failed",
  "details": [
    {
      "field": "body.text",
      "message": "String should have at least 10 characters",
      "type": "string_too_short"
    }
  ],
  "timestamp": "2026-02-19T10:30:00.000Z"
}
```

## License

MIT License

