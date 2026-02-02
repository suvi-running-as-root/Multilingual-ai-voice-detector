import base64
import io
import librosa
import numpy as np
import requests
import soundfile as sf
import torch
from fastapi import HTTPException

def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """
    Decodes audio bytes (MP3, WAV, etc.) and resamples to target_sr.
    Returns a numpy array of shape (N,).
    """
    try:
        # Load audio using soundfile or librosa (librosa handles more formats transparently but sf is faster for some)
        # using soundfile to read from memory buffer
        # loading into librosa for easy resampling if needed, but librosa.load supports file-like objects
        
        # librosa.load returns (y, sr)
        # It automatically resamples if sr is provided which is convenient
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
        return y
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {str(e)}")

def decode_base64_audio(b64_string: str) -> bytes:
    try:
        # Handle data URI scheme if present (e.g. "data:audio/mp3;base64,...")
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        
        return base64.b64decode(b64_string)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 string: {str(e)}")

def download_audio_from_url(url: str) -> bytes:
    try:
        # Wikimedia and others require a User-Agent
        headers = {"User-Agent": "VoiceDetectorAPI/1.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {str(e)}")

def process_audio_input(audio_base64: str | None, audio_url: str | None) -> np.ndarray:
    """
    Orchestrates loading audio from either base64 or URL.
    Returns the raw waveform as a numpy array at 16kHz.
    """
    if audio_base64:
        audio_bytes = decode_base64_audio(audio_base64)
    elif audio_url:
        audio_bytes = download_audio_from_url(audio_url)
    else:
        raise HTTPException(status_code=400, detail="No audio provided")
    
    return load_audio_from_bytes(audio_bytes)
