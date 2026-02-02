# 🛡️ SecureCall - AI Voice Fraud Detector

**SecureCall** is a cybersecurity API designed to detect **AI-generated voice scams** in real-time. It uses a **Dual-Model Approach** to analyze both *how* someone speaks (Audio) and *what* they say (Content).

---

## 🧠 How It Works (The Architecture)

The system passes every audio file through **two parallel AI engines**:

### 1. 🎙️ Primary Engine: AI Voice Detection
*   **Model**: `facebook/wav2vec2-large-xlsr-53`
*   **Logic**: Analyzes the raw audio signal.
*   **What it detects**:
    *   **Artificial Smoothness**: AI voices are mathematically "smoother" than real vocal cords.
    *   **Low Variance**: AI lacks the natural emotional micro-jitters of humans.
*   **Outcome**: Determines if the speaker is **HUMAN** or **AI**.

### 2. 🛡️ Secondary Engine: Content Risk Analysis
*   **Model**: `openai/whisper-small` (Multilingual)
*   **Logic**:
    1.  **Translates** audio from Hindi/Tamil/Telugu/Malayalam -> **English**.
    2.  **Scans** the English text for Fraud Keywords (e.g., "OTP", "Bank", "Police", "Money").
*   **Outcome**: Assigns a **Fraud Risk Level** (Low/Medium/High).

### 🚨 Final Verdict
The system combines both engines to give a final verdict:
*   **CRITICAL**: AI Voice + Asking for OTP.
*   **WARNING**: AI Voice (but just talking) OR Human asking for OTP.
*   **SAFE**: Human voice talking normally.

---

## 🚀 Setup & Run

### Prerequisites
1.  **Python 3.10+**
2.  **FFmpeg** (Required for audio processing)

### 🍎 Mac / Linux Setup
1.  **Install FFmpeg**:
    ```bash
    brew install ffmpeg  # Mac
    # OR
    sudo apt install ffmpeg # Linux
    ```

2.  **Setup Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Run Server**:
    ```bash
    uvicorn app.main:app --reload
    ```

### 🪟 Windows Setup (For Team)
1.  **Install FFmpeg**:
    *   Run `winget install "FFmpeg (Essentials)"` in PowerShell.
    *   *Or download manually from ffmpeg.org and add to PATH.*

2.  **Setup Environment**:
    Open **PowerShell** as Administrator:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
    *(If you get a Red Error about scripts, run: `Set-ExecutionPolicy Unrestricted -Scope Process`)*

3.  **Run Server**:
    ```powershell
    uvicorn app.main:app --reload
    ```

---

## 🧪 How to Test

### 1. Verification Script (Recommended)
We have a built-in testing tool.

**Run on Mac/Linux:**
```bash
python3 test_api.py
```

**Run on Windows:**
```powershell
python test_api.py
```

*To test your own file, open `test_api.py` in VS Code and change the file path at the bottom.*

### 2. API Response Implementation
Your frontend will receive this exact JSON format:

```json
{
  "classification": "AI",
  "confidence": 0.92,
  "explanation": "High temporal smoothness (robotic consistency)",
  "fraud_risk": "HIGH",
  "risk_keywords": ["otp", "bank"],
  "overall_risk": "CRITICAL",
  "transcript_preview": "Hello, I need your bank OTP immediately."
}
```
