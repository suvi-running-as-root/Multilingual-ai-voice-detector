class SessionContext:
    def __init__(self):
        self.chunk_ai_probs = []
        self.total_duration = 0.0
        self.num_chunks = 0
        self.escalation_flag = False

def chunk_audio_array(audio_array, metadata, chunk_duration_sec=5.0):
    sr = metadata.get("sample_rate", 16000)
    chunk_size = int(sr * chunk_duration_sec)

    for i in range(0, len(audio_array), chunk_size):
        chunk = audio_array[i:i + chunk_size]
        if len(chunk) < sr:  # skip <1s chunks
            continue
        yield chunk, metadata


from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List
from app.api.auth import verify_api_key
from app.utils.audio import process_audio_input
from app.models.detector import get_detector

router = APIRouter()

class DetectRequest(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    transcript: Optional[str] = None 
    message: Optional[str] = None

# Hackathon Exact Format
class DetectResponse(BaseModel):
    classification: str       # "AI" | "Human"
    confidence_score: float   # 0.0 - 1.0
    ai_probability: float     # Raw AI probability
    detected_language: str    # e.g. "en", "hi", "ta"
    transcription: str        # Original transcription
    english_translation: str  # English translation
    fraud_keywords: List[str] # List of detected fraud keywords
    overall_risk: str         # "HIGH" | "MEDIUM" | "LOW"
    explanation: str          # Explanation string
    # Diagnostics
    audio_duration_seconds: float       # Duration of processed audio
    num_chunks_processed: int           # Number of 30s chunks
    chunk_ai_probabilities: List[float] # AI probability per chunk
    # Deep Diagnostics
    pitch_human_score: Optional[float] = 0.0
    pitch_std: Optional[float] = 0.0
    pitch_jitter: Optional[float] = 0.0
    smoothness_score: Optional[float] = 0.0
    variance_score: Optional[float] = 0.0
    snr_score: Optional[float] = 0.0
    heuristic_score: Optional[float] = 0.0
    debug_probs: Optional[List[float]] = []
    debug_labels: Optional[dict] = {}

@router.post("/detect", response_model=DetectResponse, dependencies=[Depends(verify_api_key)])
async def detect_voice(request: DetectRequest):
    if not request.audio_base64 and not request.audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Must provide either 'audio_base64' or 'audio_url'"
        )
    
    try:
        audio_array, metadata = process_audio_input(
            request.audio_base64, request.audio_url, max_duration=6.0
        )
        
        session = SessionContext()  # Initialize session context for this request
        chunks = list(chunk_audio_array(audio_array, metadata))

        
        if audio_array is None or (hasattr(audio_array, "size") and audio_array.size == 0):
            raise HTTPException(status_code=400, detail="Audio decode produced no samples")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

    try:
        detector = get_detector()
        last_result = None

        for chunk_audio, chunk_meta in chunks:
         result = detector.detect_fraud(chunk_audio, chunk_meta)
         session.chunk_ai_probs.append(result.get("ai_probability", 0.0))
         session.total_duration += result.get("audio_duration_seconds", 0.0)
         session.num_chunks += 1
         last_result = result
         
        if session.total_duration < 2.0:
             last_result["overall_risk"] = "LOW"
             last_result["explanation"] = "Call too short to analyze reliably"
             last_result["confidence_score"] = min(last_result.get("confidence_score", 0.5), 0.4)
             
        avg_ai_prob = (sum(session.chunk_ai_probs) / len(session.chunk_ai_probs)
                       if session.chunk_ai_probs else 0.0)

        if avg_ai_prob < 0.3:
            risk = "LOW"
        elif avg_ai_prob < 0.6:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        last_result["overall_risk"] = risk
        
        if len(session.chunk_ai_probs) >= 2:
          if session.chunk_ai_probs[-1] - session.chunk_ai_probs[0] > 0.3:
             last_result["overall_risk"] = "HIGH"
             last_result["explanation"] = "Suspicious behavior increased during the call"
             
        last_result["num_chunks_processed"] = session.num_chunks
        last_result["chunk_ai_probabilities"] = session.chunk_ai_probs
        
        if "explanation" not in last_result or not last_result["explanation"]:
          last_result["explanation"] = "Voice patterns analyzed with no strong scam indicators"

        
        return last_result





    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/detect", response_model=DetectResponse, dependencies=[Depends(verify_api_key)])
async def detect_get(audio_url: str):
    """
    GET handler for Hackathon Tester. 
    Wraps the POST logic.
    """
    # Create request object
    request = DetectRequest(audio_url=audio_url)
    # Call the existing logic (we can call the service directly or the function)
    # Calling the service logic directly ensures cleaner execution stack
    return await detect_voice(request)

# --- Strict Hackathon Specification ---

class HackathonRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

class HackathonResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

from fastapi.responses import JSONResponse

@router.post("/api/voice-detection", response_model=HackathonResponse, dependencies=[Depends(verify_api_key)])
async def detect_voice_strict(request: HackathonRequest):
    """
    Strict endpoint for Hackathon evaluation.
    Path: /api/voice-detection
    """
    # 1. format check
    if request.audioFormat.lower() != "mp3":
        return JSONResponse(
            status_code=400, 
            content={"status": "error", "message": "Only mp3 format supported"}
        )

    try:
        # 2. process audio
        # Reuse existing logic via wrapper or direct call
        # process_audio_input expects (audio_base64, audio_url)
        # It handles base64 decoding.
        
        # NOTE: process_audio_input returns (numpy array, metadata)
        # OPTIMIZATION: Decode ONLY 6 seconds max to prevent timeouts on large files
        audio_array, metadata = process_audio_input(request.audioBase64, None, max_duration=2.0)
        if audio_array is None or (hasattr(audio_array, "size") and audio_array.size == 0):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Audio decode produced no samples"},
            )
        
        # 3. Detect
        detector = get_detector()
        result = detector.detect_fraud(audio_array, metadata)
        
        # 4. Map Result
        # result: {"classification": "AI"|"Human", "confidence_score": 0.xx, "explanation": "..."}
        
        mapping = {"AI": "AI_GENERATED", "Human": "HUMAN"}
        final_class = mapping.get(result.get("classification"), "HUMAN")
        
        return {
            "status": "success",
            "language": request.language,
            "classification": final_class,
            "confidenceScore": result.get("confidence_score", 0.0),
            "explanation": result.get("explanation", "Analysis completed")
        }

    except HTTPException as he:
        # Re-wrap HTTP exceptions to strict JSON format
        return JSONResponse(
            status_code=he.status_code,
            content={"status": "error", "message": he.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Server error: {str(e)}"}
        )
