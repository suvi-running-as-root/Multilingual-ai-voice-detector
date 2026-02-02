import requests
import sys
import base64

API_URL = "http://localhost:8000"
API_KEY = "demo_key_123"

def test_detect_url():
    print("\nTesting /detect (Auto-Transcribe Security Mode)...")
    payload = {
        # Using a sample file (Test file doesn't have fraud words, so transcript should be benign)
        #"audio_url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg",
        # NOTE: No "transcript" field sent! The server must generate it.
    }
    headers = {
        "X-API-Key": API_KEY,
        "User-Agent": "TestScript"
    }
    
    try:
        r = requests.post(f"{API_URL}/detect", json=payload, headers=headers)
        if r.status_code == 200:
            data = r.json()
            print("✅ Analysis Complete:")
            print(f"   Threat Level: {data['threat_level']}")
            print(f"   Is Fraud: {data['is_fraud']}")
            print(f"   Alert: {data['alert']}")
            print(f"   📝 Transcript (Heard): \"{data['transcript_preview']}\"")
            print(f"   Signals:")
            print(f"     - Voice: {data['analysis']['voice_type']}")
            print(f"     - Sentiment: {data['analysis']['sentiment']}")
            print(f"     - Bad Keywords: {data['analysis']['keywords_detected']}")
        else:
            print(f"❌ Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_detect_local_file(file_path):
    print(f"\nTesting /detect (Local File: {file_path})...")
    
    try:
        with open(file_path, "rb") as f:
            audio_content = f.read()
            encoded_string = base64.b64encode(audio_content).decode('utf-8')
            
        payload = {
            "audio_base64": encoded_string
        }
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return

    headers = {
        "X-API-Key": API_KEY,
        "User-Agent": "TestScript"
    }
    
    try:
        r = requests.post(f"{API_URL}/detect", json=payload, headers=headers)
        if r.status_code == 200:
            data = r.json()
            print("✅ Analysis Complete:")
            print(f"   Threat Level: {data['threat_level']}")
            print(f"   Is Fraud: {data['is_fraud']}")
            print(f"   Alert: {data['alert']}")
            print(f"   📝 Transcript (Heard): \"{data['transcript_preview']}\"")
            print(f"   Signals:")
            print(f"     - Voice: {data['analysis']['voice_type']}")
            print(f"     - Sentiment: {data['analysis']['sentiment']}")
            print(f"     - Bad Keywords: {data['analysis']['keywords_detected']}")
        else:
            print(f"❌ Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Uncomment the line below to test a specific local file:
    test_detect_local_file("/Users/rishitguha/Downloads/ElevenLabs_2026-02-01T13_17_32_Vanishree - Energetic Indian English_pvc_sp100_s50_sb75_se0_b_m2.mp3")
    
    # Defaults to URL test if no local file used
    test_detect_url()
