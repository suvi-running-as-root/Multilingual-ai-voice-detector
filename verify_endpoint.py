"""
Quick Endpoint Verification Script
Verifies that the /detect endpoint matches evaluation format
"""
import requests
import base64
import json

API_URL = "http://localhost:8000"
API_KEY = "demo_key_123"

def verify_endpoint():
    """Verify that /detect endpoint returns correct format"""
    print("\n" + "="*60)
    print("  ENDPOINT VERIFICATION")
    print("="*60)

    # Create a minimal test payload (1 second of silence)
    # This is just to verify the endpoint format, not accuracy
    import numpy as np
    import io
    import wave

    # Generate 1 second of silence
    sample_rate = 16000
    duration = 1
    silence = np.zeros(sample_rate * duration, dtype=np.int16)

    # Save to bytes
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence.tobytes())

    wav_bytes = wav_io.getvalue()
    audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

    # Create request
    request_payload = {
        "language": "en",
        "audioFormat": "wav",
        "audioBase64": audio_base64
    }

    print("\n1. Testing Endpoint Format")
    print("-" * 60)
    print(f"URL: {API_URL}/detect")
    print(f"Method: POST")
    print(f"Headers: X-API-Key")
    print(f"Request Fields: {list(request_payload.keys())}")

    # Send request
    try:
        response = requests.post(
            f"{API_URL}/detect",
            json=request_payload,
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            timeout=30
        )

        print(f"\n2. Response Status")
        print("-" * 60)
        print(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ FAILED: Expected 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return False

        print("✅ Status: 200 OK")

        # Parse response
        result = response.json()

        print(f"\n3. Response Format Validation")
        print("-" * 60)

        # Required fields
        required_fields = ["status", "classification", "confidenceScore"]
        has_all_required = all(field in result for field in required_fields)

        print(f"Required Fields: {required_fields}")
        print(f"Actual Fields: {list(result.keys())}")
        print(f"Has All Required: {'✅ YES' if has_all_required else '❌ NO'}")

        # Check for extra fields (should be ONLY 3 fields)
        extra_fields = [f for f in result.keys() if f not in required_fields]
        if extra_fields:
            print(f"⚠️  WARNING: Extra fields found: {extra_fields}")
            print(f"   Evaluation may fail if extra fields are present!")
        else:
            print(f"✅ Correct: ONLY 3 fields (no extras)")

        # Validate field types
        print(f"\n4. Field Value Validation")
        print("-" * 60)

        # Status
        status_valid = result.get("status") in ["success", "error"]
        print(f"status: '{result.get('status')}' - {'✅ VALID' if status_valid else '❌ INVALID (must be success/error)'}")

        # Classification
        classification = result.get("classification")
        classification_valid = classification in ["HUMAN", "AI_GENERATED"]
        print(f"classification: '{classification}' - {'✅ VALID' if classification_valid else '❌ INVALID (must be HUMAN or AI_GENERATED)'}")

        # Confidence Score
        confidence = result.get("confidenceScore")
        confidence_valid = isinstance(confidence, (int, float)) and 0 <= confidence <= 1
        print(f"confidenceScore: {confidence} - {'✅ VALID' if confidence_valid else '❌ INVALID (must be 0.0-1.0)'}")

        # Overall validation
        print(f"\n5. Overall Validation")
        print("-" * 60)

        all_valid = (
            has_all_required and
            len(extra_fields) == 0 and
            status_valid and
            classification_valid and
            confidence_valid
        )

        if all_valid:
            print("✅ PASSED: Endpoint format is CORRECT for evaluation")
            print(f"\nExample Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print("❌ FAILED: Endpoint format has issues")
            print(f"\nIssues to fix:")
            if not has_all_required:
                print(f"  - Missing required fields")
            if extra_fields:
                print(f"  - Remove extra fields: {extra_fields}")
            if not status_valid:
                print(f"  - Fix status value")
            if not classification_valid:
                print(f"  - Fix classification value (must be HUMAN or AI_GENERATED)")
            if not confidence_valid:
                print(f"  - Fix confidenceScore (must be 0.0-1.0)")
            return False

    except requests.exceptions.Timeout:
        print("❌ FAILED: Request timed out (>30 seconds)")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Cannot connect to server")
        print(f"   Is the server running at {API_URL}?")
        return False
    except Exception as e:
        print(f"❌ FAILED: Error - {e}")
        return False

def check_health():
    """Check if server is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
    except:
        pass
    print("❌ Server is NOT running")
    print(f"   Start with: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
    return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI Voice Detection API - Endpoint Verification")
    print("="*60)

    # Check health first
    print("\nStep 1: Check Server Health")
    print("-" * 60)
    if not check_health():
        print("\nPlease start the server first!")
        exit(1)

    # Verify endpoint
    print("\nStep 2: Verify /detect Endpoint Format")
    if verify_endpoint():
        print("\n" + "="*60)
        print("  ✅ READY FOR EVALUATION")
        print("="*60)
        print("\nNext Steps:")
        print("  1. Run full tests: python test_evaluation.py")
        print("  2. Deploy for evaluation")
        print("  3. Target: 100/100 points")
    else:
        print("\n" + "="*60)
        print("  ❌ FIX ISSUES BEFORE EVALUATION")
        print("="*60)
        print("\nPlease fix the issues above before deploying")
