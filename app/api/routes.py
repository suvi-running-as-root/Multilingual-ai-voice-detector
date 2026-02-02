from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from app.api.auth import verify_api_key
from app.utils.audio import process_audio_input
from app.models.detector import get_detector

router = APIRouter()

class DetectRequest(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    transcript: Optional[str] = None # Optional override, otherwise auto-generated
    message: Optional[str] = None

class AnalysisDetail(BaseModel):
    voice_type: str
    sentiment: str
    keywords_detected: List[str]

class DetectResponse(BaseModel):
    threat_level: str
    is_fraud: bool
    alert: str
    transcript_preview: str # Returning what the model heard
    analysis: AnalysisDetail

@router.post("/detect", response_model=DetectResponse, dependencies=[Depends(verify_api_key)])
async def detect_voice(request: DetectRequest):
    if not request.audio_base64 and not request.audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Must provide either 'audio_base64' or 'audio_url'"
        )
    
    try:
        audio_array = process_audio_input(request.audio_base64, request.audio_url)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

    try:
        detector = get_detector()
        # Pass transcript only if user forced one, otherwise let model generate it
        result = detector.detect_fraud(audio_array, provided_transcript=request.transcript)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
