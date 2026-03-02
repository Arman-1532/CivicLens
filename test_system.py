#!/usr/bin/env python3
"""
Comprehensive System Test for CivicLens
Tests Backend, ML Pipeline, and generates test report
"""

import sys
import os
from pathlib import Path

# Add paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "backend"))
sys.path.insert(0, str(BASE_DIR / "ml"))

def test_backend():
    """Test backend components"""
    print("\n" + "=" * 60)
    print("BACKEND TESTS")
    print("=" * 60)

    results = []

    # Test 1: Config
    try:
        from app.core.config import settings
        assert settings.APP_NAME == "CivicLens"
        print(f"✅ Config: {settings.APP_NAME} v{settings.APP_VERSION}")
        results.append(("Config", True))
    except Exception as e:
        print(f"❌ Config: {e}")
        results.append(("Config", False))

    # Test 2: Preprocessing
    try:
        from app.services.preprocessing import preprocess_text
        text = "The WATER supply has been IRREGULAR!!! Call 1234567890"
        cleaned = preprocess_text(text)
        assert "water" in cleaned
        assert "1234567890" not in cleaned
        print(f"✅ Preprocessing: '{text[:30]}...' -> '{cleaned[:30]}...'")
        results.append(("Preprocessing", True))
    except Exception as e:
        print(f"❌ Preprocessing: {e}")
        results.append(("Preprocessing", False))

    # Test 3: Category Mapping
    try:
        from app.models.complaint_model import get_department_for_category, VALID_CATEGORIES
        dept = get_department_for_category("Corruption")
        assert dept == "Anti-Corruption Bureau"
        print(f"✅ Category Mapping: Corruption -> {dept}")
        results.append(("Category Mapping", True))
    except Exception as e:
        print(f"❌ Category Mapping: {e}")
        results.append(("Category Mapping", False))

    # Test 4: Urgency Determination
    try:
        from app.models.complaint_model import determine_urgency
        urgency = determine_urgency("This is an emergency!", "Service Delay")
        assert urgency == "High"
        print(f"✅ Urgency: 'emergency' text -> {urgency}")
        results.append(("Urgency", True))
    except Exception as e:
        print(f"❌ Urgency: {e}")
        results.append(("Urgency", False))

    # Test 5: Pydantic Schemas
    try:
        from app.models.schema import ComplaintRequest, PredictionResponse
        from datetime import datetime
        req = ComplaintRequest(text="This is a valid test complaint")
        resp = PredictionResponse(
            category="Utility Issue",
            department="Public Utilities",
            urgency="Medium",
            confidence=0.85,
            timestamp=datetime.utcnow()
        )
        print(f"✅ Schemas: ComplaintRequest & PredictionResponse working")
        results.append(("Schemas", True))
    except Exception as e:
        print(f"❌ Schemas: {e}")
        results.append(("Schemas", False))

    # Test 6: Prediction Service
    try:
        from app.services.prediction_service import get_prediction_service
        service = get_prediction_service()
        loaded = service.load_models()
        if loaded:
            print(f"✅ Prediction Service: Models loaded successfully")
            results.append(("Prediction Service", True))
        else:
            print(f"⚠️ Prediction Service: Models not loaded (need to train first)")
            results.append(("Prediction Service", False))
    except Exception as e:
        print(f"❌ Prediction Service: {e}")
        results.append(("Prediction Service", False))

    # Test 7: FastAPI App
    try:
        from app.main import app
        assert app.title == "CivicLens"
        routes = [r.path for r in app.routes]
        assert "/" in routes
        print(f"✅ FastAPI App: {app.title} with {len(routes)} routes")
        results.append(("FastAPI App", True))
    except Exception as e:
        print(f"❌ FastAPI App: {e}")
        results.append(("FastAPI App", False))

    return results


def test_ml_pipeline():
    """Test ML pipeline components"""
    print("\n" + "=" * 60)
    print("ML PIPELINE TESTS")
    print("=" * 60)

    results = []

    # Test 1: Clean Data
    try:
        from data_processing.clean_data import clean_text
        text = "Email test@example.com for HELP!!!"
        cleaned = clean_text(text)
        assert "test@example.com" not in cleaned
        assert "help" in cleaned
        print(f"✅ Clean Data: Text cleaning working")
        results.append(("Clean Data", True))
    except Exception as e:
        print(f"❌ Clean Data: {e}")
        results.append(("Clean Data", False))

    # Test 2: Label Mapping
    try:
        from data_processing.label_mapping import map_label, get_unified_categories
        mapped = map_label("water supply")
        assert mapped == "Utility Issue"
        cats = get_unified_categories()
        assert len(cats) == 6
        print(f"✅ Label Mapping: 'water supply' -> '{mapped}', {len(cats)} categories")
        results.append(("Label Mapping", True))
    except Exception as e:
        print(f"❌ Label Mapping: {e}")
        results.append(("Label Mapping", False))

    # Test 3: Synthetic Dataset
    try:
        from data_processing.merge_datasets import create_synthetic_dataset
        df = create_synthetic_dataset(samples_per_category=10)
        assert len(df) == 60  # 6 categories * 10 samples
        assert 'text' in df.columns
        assert 'category' in df.columns
        print(f"✅ Synthetic Dataset: {len(df)} samples created")
        results.append(("Synthetic Dataset", True))
    except Exception as e:
        print(f"❌ Synthetic Dataset: {e}")
        results.append(("Synthetic Dataset", False))

    # Test 4: Model Files Exist
    try:
        model_path = BASE_DIR / "backend" / "trained_models" / "classifier.pkl"
        vectorizer_path = BASE_DIR / "backend" / "trained_models" / "tfidf_vectorizer.pkl"

        model_exists = model_path.exists() and model_path.stat().st_size > 100
        vectorizer_exists = vectorizer_path.exists() and vectorizer_path.stat().st_size > 100

        if model_exists and vectorizer_exists:
            print(f"✅ Model Files: classifier.pkl and tfidf_vectorizer.pkl exist")
            results.append(("Model Files", True))
        else:
            print(f"⚠️ Model Files: Missing or empty (run training first)")
            results.append(("Model Files", False))
    except Exception as e:
        print(f"❌ Model Files: {e}")
        results.append(("Model Files", False))

    return results


def test_prediction():
    """Test end-to-end prediction"""
    print("\n" + "=" * 60)
    print("END-TO-END PREDICTION TEST")
    print("=" * 60)

    results = []

    try:
        from app.services.prediction_service import get_prediction_service

        service = get_prediction_service()
        if not service.load_models():
            print("⚠️ Models not loaded - skipping prediction test")
            return [("E2E Prediction", False)]

        test_cases = [
            ("The water supply has been irregular for a week", "Utility Issue"),
            ("Government official asked for bribe", "Corruption"),
            ("Police not responding to complaints", "Law Enforcement Issue"),
            ("Electricity bill shows wrong charges", "Financial Issue"),
            ("Passport application pending for 6 months", "Service Delay"),
            ("Facing harassment at workplace", "Harassment"),
        ]

        correct = 0
        for text, expected in test_cases:
            result = service.predict(text)
            is_correct = result['category'] == expected
            if is_correct:
                correct += 1
            status = "✅" if is_correct else "❌"
            print(f"  {status} '{text[:40]}...'")
            print(f"      Expected: {expected}, Got: {result['category']} ({result['confidence']:.1%})")

        accuracy = correct / len(test_cases)
        print(f"\n✅ Prediction Accuracy: {correct}/{len(test_cases)} ({accuracy:.0%})")
        results.append(("E2E Prediction", accuracy >= 0.5))

    except Exception as e:
        print(f"❌ E2E Prediction: {e}")
        results.append(("E2E Prediction", False))

    return results


def test_frontend_files():
    """Test frontend files exist"""
    print("\n" + "=" * 60)
    print("FRONTEND FILES TEST")
    print("=" * 60)

    results = []
    frontend_dir = BASE_DIR / "frontend"

    required_files = [
        "package.json",
        "vite.config.js",
        "tailwind.config.js",
        "index.html",
        "src/App.jsx",
        "src/main.jsx",
        "src/index.css",
        "src/services/api.js",
        "src/components/ComplaintForm.jsx",
        "src/components/ResultCard.jsx",
        "src/components/Navbar.jsx",
        "src/pages/Home.jsx",
        "src/pages/Dashboard.jsx",
        "src/pages/AdminPanel.jsx",
        "src/context/ComplaintContext.jsx",
    ]

    missing = []
    for f in required_files:
        path = frontend_dir / f
        if not path.exists():
            missing.append(f)

    if not missing:
        print(f"✅ All {len(required_files)} frontend files exist")
        results.append(("Frontend Files", True))
    else:
        print(f"❌ Missing files: {missing}")
        results.append(("Frontend Files", False))

    # Check node_modules
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists():
        print(f"✅ node_modules installed")
        results.append(("Node Modules", True))
    else:
        print(f"⚠️ node_modules not found (run 'npm install')")
        results.append(("Node Modules", False))

    return results


def main():
    print("=" * 60)
    print("CIVICLENS COMPREHENSIVE SYSTEM TEST")
    print("=" * 60)

    all_results = []

    # Run all tests
    all_results.extend(test_backend())
    all_results.extend(test_ml_pipeline())
    all_results.extend(test_prediction())
    all_results.extend(test_frontend_files())

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status in all_results if status)
    total = len(all_results)

    for name, status in all_results:
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 60)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready.")
    elif passed >= total * 0.7:
        print("\n⚠️ Most tests passed. Check failed tests above.")
    else:
        print("\n❌ Multiple tests failed. Please review the output.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

