"""
Test script for evaluation-compliant /detect endpoint
This script tests the exact format required by the evaluation system
"""
import requests
import base64
import json
import os
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "demo_key_123"

def test_detect_endpoint(audio_file_path, language="en", expected_classification=None):
    """
    Test the /detect endpoint with exact evaluation format

    Args:
        audio_file_path: Path to audio file
        language: Language code (en, hi, ta, te, ml)
        expected_classification: Expected result for validation (HUMAN or AI_GENERATED)
    """
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(audio_file_path)}")
    print(f"{'='*60}")

    # Read and encode audio file
    try:
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    except FileNotFoundError:
        print(f"❌ File not found: {audio_file_path}")
        return None

    # Determine audio format from file extension
    ext = Path(audio_file_path).suffix.lower()
    audio_format = {
        '.mp3': 'mp3',
        '.mpeg': 'mpeg',
        '.wav': 'wav',
        '.ogg': 'ogg'
    }.get(ext, 'mp3')

    # Create request in EXACT evaluation format
    request_payload = {
        "language": language,
        "audioFormat": audio_format,
        "audioBase64": audio_base64
    }

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    print(f"Request: language={language}, audioFormat={audio_format}")
    print(f"Audio size: {len(audio_bytes)} bytes ({len(audio_bytes)/1024:.1f} KB)")

    # Send request
    try:
        response = requests.post(
            f"{API_URL}/detect",
            json=request_payload,
            headers=headers,
            timeout=30
        )

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # Verify response format
            required_fields = ["status", "classification", "confidenceScore"]
            has_all_fields = all(field in result for field in required_fields)

            print(f"\n📊 RESPONSE:")
            print(json.dumps(result, indent=2))

            print(f"\n✅ Validation:")
            print(f"   - Has all required fields: {has_all_fields}")
            print(f"   - Status: {result.get('status')}")
            print(f"   - Classification: {result.get('classification')}")
            print(f"   - Confidence: {result.get('confidenceScore')}")

            # Check if classification is correct
            if expected_classification:
                is_correct = result.get('classification') == expected_classification
                print(f"   - Expected: {expected_classification}")
                print(f"   - Match: {'✅ CORRECT' if is_correct else '❌ WRONG'}")

                # Calculate points (each correct = 25 points)
                points = 25 if is_correct else 0
                print(f"   - Points: {points}/25")

                return {
                    "correct": is_correct,
                    "points": points,
                    "confidence": result.get('confidenceScore', 0),
                    "result": result
                }

            return result

        else:
            print(f"❌ Error Response:")
            print(response.text)
            return None

    except requests.exceptions.Timeout:
        print(f"❌ Request timed out (>30 seconds)")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def run_evaluation_tests():
    """
    Run complete evaluation test suite
    Tests all available audio files
    """
    print("\n" + "="*60)
    print("AI VOICE DETECTION API - EVALUATION TEST SUITE")
    print("="*60)

    # Define test cases with expected classifications
    test_cases = [
        # English samples
        ("English_voice_AI_GENERATED.mp3.mpeg", "en", "AI_GENERATED"),
        ("ai_english_fraud.mp3", "en", "AI_GENERATED"),
        ("ai_english_real.mp3", "en", "AI_GENERATED"),

        # Hindi samples
        ("Hindi_Voice_HUMAN.mp3.mpeg", "hi", "HUMAN"),
        ("ai_hindi_fraud.mp3", "hi", "AI_GENERATED"),

        # Other languages
        ("Malayalam_AI_GENERATED.mp3.mpeg", "ml", "AI_GENERATED"),
        ("TAMIL_VOICE__HUMAN.mp3.mpeg", "ta", "HUMAN"),
        ("Telugu_Voice_AI_GENERATED.mp3.mpeg", "te", "AI_GENERATED"),
    ]

    results = []
    total_points = 0
    max_points = 0

    for audio_file, language, expected in test_cases:
        audio_path = os.path.join(os.getcwd(), audio_file)

        if os.path.exists(audio_path):
            result = test_detect_endpoint(audio_path, language, expected)
            if result and isinstance(result, dict) and 'points' in result:
                results.append(result)
                total_points += result['points']
                max_points += 25
        else:
            print(f"\n⚠️  Skipping {audio_file} (file not found)")

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(results)}")
    print(f"Correct: {sum(1 for r in results if r['correct'])}")
    print(f"Wrong: {sum(1 for r in results if not r['correct'])}")
    print(f"Score: {total_points}/{max_points} points")
    print(f"Accuracy: {total_points/max_points*100:.1f}%" if max_points > 0 else "N/A")

    # Confidence analysis
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"Average Confidence: {avg_confidence:.2f}")
        print(f"Min Confidence: {min(r['confidence'] for r in results):.2f}")
        print(f"Max Confidence: {max(r['confidence'] for r in results):.2f}")

    print("="*60)

    return results

def test_single_file(file_path):
    """Test a single file"""
    test_detect_endpoint(file_path, language="en", expected_classification=None)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test specific file
        test_single_file(sys.argv[1])
    else:
        # Run full evaluation suite
        run_evaluation_tests()
