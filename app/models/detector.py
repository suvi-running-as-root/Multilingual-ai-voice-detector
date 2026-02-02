import torch
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, pipeline

class VoiceDetector:
    _instance = None
    
    def __init__(self):
        # 1. Load Audio Feature Model (Web2Vec2 for Deepfake detection)
        self.model_name = "facebook/wav2vec2-large-xlsr-53"
        print(f"Loading Security Model: {self.model_name} ...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.model.eval()
        
        # 2. Load Speech-to-Text Model (Whisper for Transcript)
        print("Loading Whisper Model for Transcription...")
        # Using "openai/whisper-tiny.en" for speed and English focus. 
        # Remove ".en" if multilingual needed, but tiny is best for speed.
        self.transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
        print("All Models loaded successfully.")
        
        # Simulated "Kaggle Fraud Dataset" - Common trigger words
        self.fraud_keywords = {
            "financial": ["bank", "account", "credit card", "cvv", "pin", "verify", "blocked", "suspended"],
            "urgency": ["immediately", "urgent", "expires", "arrest", "police", "legal action", "warrant"],
            "scams": ["lottery", "winner", "refund", "gift card", "tech support", "virus", "hacked"]
        }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _analyze_audio_sentiment(self, audio_array: np.ndarray):
        """
        Heuristic for Voice Sentiment/Urgency using signal processing.
        """
        # 1. Energy (RMS)
        rms = librosa.feature.rms(y=audio_array).mean()
        
        # 2. Spectral Centroid (associated with "brightness" or aggression)
        cent = librosa.feature.spectral_centroid(y=audio_array, sr=16000).mean()
        
        sentiment = "Neutral"
        urgency_score = 0.0
        
        if rms > 0.05 or cent > 2500:
            sentiment = "Urgent/Aggressive"
            urgency_score = 0.8
        elif rms < 0.005:
            sentiment = "Whisper/Quiet"
            urgency_score = 0.1
            
        return sentiment, urgency_score

    def detect_fraud(self, audio_array: np.ndarray, provided_transcript: str = None):
        """
        Cybersecurity Logic with Auto-Transcription & Sentiment
        """
        # --- PHASE 1: TRANSCRIPTION ---
        # If no transcript provided, generate one from audio
        final_transcript = provided_transcript
        if not final_transcript:
            try:
                # Whisper pipeline expects raw audio or file path. 
                # We can pass the numpy array directly since we sampled at 16k which whisper generally handles 
                # (though usually expects 16k).
                # Note: Pipeline implementation handles some resampling but good to ensure compatible input.
                result = self.transcriber(audio_array)
                final_transcript = result.get("text", "")
            except Exception as e:
                print(f"Transcription Warning: {e}")
                final_transcript = ""

        # --- PHASE 2: DEEPFAKE DETECTION ---
        inputs = self.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[0].numpy() 
        
        time_variance = np.var(embeddings, axis=0).mean()
        ai_score = 1.0 / (1.0 + time_variance * 20.0) 
        ai_score = float(np.clip(ai_score, 0.0, 1.0))
        
        # --- PHASE 3: SENTIMENT ---
        voice_sentiment, urgency_score = self._analyze_audio_sentiment(audio_array)

        # --- PHASE 4: KEYWORDS ---
        keyword_hits = []
        if final_transcript:
            text = final_transcript.lower()
            for category, words in self.fraud_keywords.items():
                for word in words:
                    if word in text:
                        keyword_hits.append(f"{category}:{word}")

        # --- PHASE 5: THREAT ASSESSMENT ---
        threat_level = "Low"
        alert = "No immediate threats detected."
        is_fraud = False

        total_risk = ai_score * 0.5 
        
        if keyword_hits:
            total_risk += 0.3
        
        if urgency_score > 0.6:
            total_risk += 0.1

        if total_risk > 0.5:
            is_fraud = True
            if total_risk > 0.7:
                threat_level = "High"
                alert = "CRITICAL: High-risk call detected."
            else:
                threat_level = "Medium"
                alert = "WARNING: Suspicious patterns detected."
        
        if ai_score > 0.6:
             alert += " Potential Deepfake."

        return {
            "threat_level": threat_level,
            "is_fraud": is_fraud,
            "alert": alert,
            "transcript_preview": final_transcript[:200] + "..." if len(final_transcript) > 200 else final_transcript,
            "analysis": {
                "voice_type": "AI" if ai_score > 0.5 else "Human",
                "sentiment": voice_sentiment,
                "keywords_detected": keyword_hits
            }
        }

# Global instance
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = VoiceDetector.get_instance()
    return detector
