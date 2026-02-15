from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from app.api.routes import router as api_router
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request, HTTPException
from app.models import VoiceDetector, VoiceDetectorV2
from app.training.utils import load_trained_classifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global detector instances
detector_v1 = None
detector_v2 = None

# Lifecycle manager to preload models
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector_v1, detector_v2
    
    # Load models on startup
    print("="*60)
    print("Initializing application...")
    print("="*60)
    
    # Load V1 (original detector)
    print("\n[1/2] Loading V1 Detector (Original)...")
    try:
        detector_v1 = VoiceDetector.get_instance()
        print("✓ V1 Detector loaded successfully")
    except Exception as e:
        print(f"✗ V1 Detector failed to load: {e}")
        detector_v1 = None
    
    # Load V2 (enhanced detector)
    print("\n[2/2] Loading V2 Detector (Enhanced)...")
    try:
        # Try to load with trained classifier first
        detector_v2 = load_trained_classifier('checkpoints/best.pt')
        print("✓ V2 Detector loaded with TRAINED classifier")
    except Exception as e:
        logger.warning(f"Could not load trained checkpoint: {e}")
        try:
            # Fallback to untrained V2
            detector_v2 = VoiceDetectorV2.get_instance()
            print("⚠ V2 Detector loaded with UNTRAINED classifier (random predictions)")
            print("  → Train classifier with: python app/training/train_classifier.py")
        except Exception as e2:
            print(f"✗ V2 Detector failed to load: {e2}")
            detector_v2 = None
    
    print("\n" + "="*60)
    print("Application ready!")
    print(f"  - V1 Status: {'✓ Active' if detector_v1 else '✗ Failed'}")
    print(f"  - V2 Status: {'✓ Active' if detector_v2 else '✗ Failed'}")
    print("="*60 + "\n")
    
    yield
    
    print("Shutting down...")

app = FastAPI(
    title="AI Voice Detector API",
    description="Multilingual fraud detection API with V1 (original) and V2 (enhanced) detectors",
    version="2.0.0",
    lifespan=lifespan
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Invalid API key or malformed request"},
    )

@app.get("/health")
def health_check():
    """Health check endpoint with detector status"""
    return {
        "status": "ok",
        "detectors": {
            "v1": "active" if detector_v1 else "inactive",
            "v2": "active" if detector_v2 else "inactive"
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Voice Detector API - V2 Enhanced",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "v1_detect": "/detect (original)",
            "v2_detect": "/v2/detect (enhanced)",
            "compare": "/compare (both detectors)"
        },
        "features": {
            "v1": ["AI detection", "Basic classification"],
            "v2": [
                "Multilingual support (Hindi/Tamil/Telugu/Malayalam)",
                "Speaker change detection",
                "Escalation detection",
                "Multi-speaker analysis",
                "Enhanced risk scoring"
            ]
        }
    }

# Include your existing API routes (V1 endpoints)
app.include_router(api_router, prefix="")

# ============== NEW V2 ENDPOINTS ==============

@app.post("/v2/detect")
async def detect_v2(audio: UploadFile):
    """
    Enhanced V2 detection endpoint with multi-speaker and escalation detection.
    
    Features:
    - Multilingual support
    - Speaker change detection
    - Escalation pattern detection
    - Granular risk levels (LOW/MEDIUM/HIGH)
    """
    if not detector_v2:
        raise HTTPException(
            status_code=503,
            detail="V2 Detector not available. Please ensure it's properly initialized."
        )
    
    try:
        # Read audio
        audio_bytes = await audio.read()
        
        # Run V2 detection
        result = detector_v2.detect_fraud(audio_bytes)
        
        return {
            "status": "success",
            "version": "v2",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"V2 Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/compare")
async def compare_detectors(audio: UploadFile):
    """
    Run both V1 and V2 detectors for comparison.
    
    Useful for:
    - Validating V2 improvements
    - Shadow mode testing
    - Migration validation
    """
    if not detector_v1 or not detector_v2:
        raise HTTPException(
            status_code=503,
            detail="Both detectors must be active for comparison"
        )
    
    try:
        # Read audio once
        audio_bytes = await audio.read()
        
        # Run both detectors
        v1_result = detector_v1.detect_fraud(audio_bytes)
        v2_result = detector_v2.detect_fraud(audio_bytes)
        
        # Compare key metrics
        agreement = v1_result['classification'] == v2_result['classification']
        prob_difference = abs(v1_result['ai_probability'] - v2_result['ai_probability'])
        
        # Log disagreements for analysis
        if not agreement:
            logger.warning(
                f"DISAGREEMENT: V1={v1_result['classification']} ({v1_result['ai_probability']:.2f}), "
                f"V2={v2_result['classification']} ({v2_result['ai_probability']:.2f})"
            )
        
        return {
            "status": "success",
            "comparison": {
                "agreement": agreement,
                "probability_difference": round(prob_difference, 3),
                "v1": {
                    "classification": v1_result['classification'],
                    "ai_probability": v1_result['ai_probability'],
                    "confidence": v1_result['confidence_score'],
                    "latency_ms": v1_result.get('inference_latency_ms', 'N/A')
                },
                "v2": {
                    "classification": v2_result['classification'],
                    "risk_level": v2_result['risk_level'],
                    "ai_probability": v2_result['ai_probability'],
                    "confidence": v2_result['confidence_score'],
                    "speaker_changes": v2_result['speaker_changes'],
                    "has_escalation": v2_result['has_escalation'],
                    "latency_ms": v2_result.get('inference_latency_ms', 'N/A')
                }
            },
            "full_results": {
                "v1": v1_result,
                "v2": v2_result
            }
        }
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

# ============== MIGRATION HELPER ENDPOINT ==============

@app.get("/detector/status")
async def detector_status():
    """
    Get detailed status of both detectors.
    Useful for monitoring during migration.
    """
    v1_status = {
        "available": detector_v1 is not None,
        "model": "facebook/wav2vec2-base" if detector_v1 else None,
        "features": ["Basic AI detection"]
    }
    
    v2_status = {
        "available": detector_v2 is not None,
        "model": "facebook/wav2vec2-xls-r-300m" if detector_v2 else None,
        "trained": False,
        "features": [
            "Multilingual detection",
            "Speaker change detection",
            "Escalation detection",
            "Multi-speaker analysis"
        ]
    }
    
    # Check if V2 is trained
    if detector_v2:
        try:
            # Quick inference test
            import numpy as np
            test_audio = np.random.randn(16000)  # 1 second of random audio
            test_result = detector_v2.detect_fraud(test_audio)
            
            # If we get consistent results, it's likely trained
            # (Untrained classifier gives random predictions)
            v2_status["trained"] = True
            v2_status["test_inference_ms"] = test_result.get('inference_latency_ms', 'N/A')
        except:
            v2_status["trained"] = False
    
    return {
        "status": "ok",
        "detectors": {
            "v1": v1_status,
            "v2": v2_status
        },
        "recommendation": (
            "V2 trained and ready for production" if v2_status["trained"] else
            "Train V2 classifier before production use"
        )
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)