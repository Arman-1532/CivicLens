#!/usr/bin/env python3
"""
Backend Test Script for CivicLens
Tests all components without running the server
"""

import sys

def test_imports():
    """Test all imports work correctly"""
    print("=== TESTING IMPORTS ===")
    try:
        from app.core.config import settings
        print(f"✓ Config loaded: {settings.APP_NAME} v{settings.APP_VERSION}")

        from app.services.preprocessing import preprocess_text
        print("✓ Preprocessing service imported")

        from app.models.complaint_model import get_department_for_category, determine_urgency, VALID_CATEGORIES
        print("✓ Complaint model imported")

        from app.models.schema import ComplaintRequest, PredictionResponse
        print("✓ Pydantic schemas imported")

        from app.services.prediction_service import get_prediction_service
        print("✓ Prediction service imported")

        from app.main import app
        print("✓ FastAPI app imported")

        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_preprocessing():
    """Test text preprocessing"""
    print("\n=== TESTING PREPROCESSING ===")
    from app.services.preprocessing import preprocess_text

    test_cases = [
        ("The water supply has been irregular!", "water supply irregular"),
        ("URGENT: Police not responding to calls", "urgent police responding calls"),
        ("Email me at test@example.com for details", "email details"),
    ]

    all_passed = True
    for original, expected_words in test_cases:
        cleaned = preprocess_text(original)
        # Check that expected words are present (order may vary)
        for word in expected_words.split():
            if word not in cleaned:
                print(f"✗ Failed: '{original}' -> '{cleaned}' (missing '{word}')")
                all_passed = False
                break
        else:
            print(f"✓ '{original[:40]}...' -> '{cleaned[:40]}...'")

    return all_passed

def test_category_mappings():
    """Test category to department mappings"""
    print("\n=== TESTING CATEGORY MAPPINGS ===")
    from app.models.complaint_model import get_department_for_category, VALID_CATEGORIES

    expected_mappings = {
        "Corruption": "Anti-Corruption Bureau",
        "Utility Issue": "Public Utilities Department",
        "Service Delay": "Administrative Services",
        "Harassment": "Women's Commission / HR",
        "Financial Issue": "Finance Department",
        "Law Enforcement Issue": "Police Department / Law Enforcement",
    }

    all_passed = True
    for category, expected_dept in expected_mappings.items():
        actual_dept = get_department_for_category(category)
        if actual_dept == expected_dept:
            print(f"✓ {category} -> {actual_dept}")
        else:
            print(f"✗ {category}: expected '{expected_dept}', got '{actual_dept}'")
            all_passed = False

    return all_passed

def test_urgency_determination():
    """Test urgency level determination"""
    print("\n=== TESTING URGENCY DETERMINATION ===")
    from app.models.complaint_model import determine_urgency

    test_cases = [
        ("This is an emergency!", "Service Delay", "High"),  # emergency keyword
        ("Urgent help needed", "Utility Issue", "High"),  # urgent keyword
        ("The service is delayed", "Service Delay", "Medium"),  # delay keyword
        ("Regular complaint text", "Corruption", "High"),  # default for corruption
        ("Regular complaint text", "Service Delay", "Low"),  # default for service delay
    ]

    all_passed = True
    for text, category, expected_urgency in test_cases:
        actual_urgency = determine_urgency(text, category)
        if actual_urgency == expected_urgency:
            print(f"✓ '{text[:30]}...' ({category}) -> {actual_urgency}")
        else:
            print(f"✗ '{text[:30]}...' ({category}): expected '{expected_urgency}', got '{actual_urgency}'")
            all_passed = False

    return all_passed

def test_pydantic_schemas():
    """Test Pydantic schema validation"""
    print("\n=== TESTING PYDANTIC SCHEMAS ===")
    from app.models.schema import ComplaintRequest, PredictionResponse
    from datetime import datetime

    all_passed = True

    # Test valid complaint request
    try:
        req = ComplaintRequest(text="This is a valid complaint with enough text")
        print(f"✓ Valid ComplaintRequest created: '{req.text[:30]}...'")
    except Exception as e:
        print(f"✗ ComplaintRequest failed: {e}")
        all_passed = False

    # Test invalid complaint request (too short)
    try:
        req = ComplaintRequest(text="Short")
        print(f"✗ Short text should have failed validation")
        all_passed = False
    except Exception as e:
        print(f"✓ Short text correctly rejected: {type(e).__name__}")

    # Test prediction response
    try:
        resp = PredictionResponse(
            category="Utility Issue",
            department="Public Utilities Department",
            urgency="Medium",
            confidence=0.87,
            timestamp=datetime.utcnow()
        )
        print(f"✓ PredictionResponse created: {resp.category} ({resp.confidence})")
    except Exception as e:
        print(f"✗ PredictionResponse failed: {e}")
        all_passed = False

    return all_passed

def test_prediction_service():
    """Test prediction service initialization"""
    print("\n=== TESTING PREDICTION SERVICE ===")
    from app.services.prediction_service import get_prediction_service

    service = get_prediction_service()
    print(f"✓ Prediction service created")
    print(f"  - Model loaded: {service.model_loaded}")
    print(f"  - Vectorizer loaded: {service.vectorizer_loaded}")
    print(f"  - Service ready: {service.is_ready}")

    if not service.is_ready:
        print("  (Note: Models not loaded - this is expected until training is complete)")

    return True

def test_fastapi_app():
    """Test FastAPI app configuration"""
    print("\n=== TESTING FASTAPI APP ===")
    from app.main import app

    print(f"✓ App title: {app.title}")
    print(f"✓ App version: {app.version}")
    print(f"✓ Docs URL: {app.docs_url}")

    # Check routes are registered
    routes = [route.path for route in app.routes]
    expected_routes = ["/", "/api/v1/health", "/api/v1/complaints/predict"]

    all_found = True
    for expected in expected_routes:
        found = any(expected in route for route in routes)
        if found:
            print(f"✓ Route registered: {expected}")
        else:
            print(f"✗ Route missing: {expected}")
            all_found = False

    return all_found

def main():
    """Run all tests"""
    print("=" * 50)
    print("CivicLens Backend Test Suite")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Preprocessing", test_preprocessing),
        ("Category Mappings", test_category_mappings),
        ("Urgency Determination", test_urgency_determination),
        ("Pydantic Schemas", test_pydantic_schemas),
        ("Prediction Service", test_prediction_service),
        ("FastAPI App", test_fastapi_app),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Backend is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

