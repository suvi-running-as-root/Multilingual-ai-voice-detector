import torch
import numpy as np
import librosa
import torch
import numpy as np
import librosa
# import noisereduce as nr (Disabled: causes artifacts)
import io
import requests
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification, pipeline
import torch.nn.functional as F

class VoiceDetector:
    _instance = None
    
    def __init__(self):
        from torch.nn import CosineSimilarity
        self.cos_sim = CosineSimilarity(dim=1, eps=1e-6)

        print("Initializing Detection Pipeline...")
        
        # 1. Primary AI vs Human detection (language-agnostic)
        # Using a multilingual XLS-R based model for better Hindi/non-English support
        self.detector_model_name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
        print(f"Loading AI Detector: {self.detector_model_name} ...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.detector_model_name
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.detector_model_name
        )
        self.model.eval()
        
        # 2. Transcription and Translation (DISABLED FOR SPEED)
        self.whisper_model_name = None 
        self.transcriber = None
        
        # 3. Fraud Keywords (DISABLED FOR SPEED)
        self.fraud_keywords = []
        
        print("AI Detector loaded successfully. (Whisper/Fraud disabled for performance)")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_audio(self, input_audio):
        """
        Download or decode the audio from URL or Base64/Bytes.
        Returns floating point audio array.
        """
        # If input is URL
        if isinstance(input_audio, str) and input_audio.startswith("http"):
            response = requests.get(input_audio)
            response.raise_for_status()
            audio_bytes = io.BytesIO(response.content)
        # If input is bytes-like (file or base64 decoded bytes)
        elif isinstance(input_audio, (bytes, bytearray, io.BytesIO)):
             if isinstance(input_audio, (bytes, bytearray)):
                 audio_bytes = io.BytesIO(input_audio)
             else:
                 audio_bytes = input_audio
        elif isinstance(input_audio, np.ndarray):
             # Already loaded audio
             return input_audio, 16000 # Assume 16k if passed from utils, or check logic
        else:
            # Assume it's a file path or direct numpy (if passed locally)
            audio_bytes = input_audio

        # Load with Librosa
        try:
             # librosa.load can handle path or file-like object
             y, sr = librosa.load(audio_bytes, sr=None)
             return y, sr
        except Exception as e:
             raise ValueError(f"Failed to load audio: {e}")

    def _preprocess_audio(self, y, sr):
        """
        Convert to mono, 16 kHz.
        Apply: noise reduction, silence trimming, normalization to -1..1.
        Return processed audio and new sample rate (16000).
        """
        target_sr = 16000
        
        # 1. Convert to mono and resample to 16kHz
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Ensure mono (librosa.load defaults to mono=True, but just in case)
        if y.ndim > 1:
            y = librosa.to_mono(y)
            
        # 2. Noise Reduction
        # Using stationary noise reduction
        # noisy reducation causing artifacts on clean audio? 
        # y = nr.reduce_noise(y=y, sr=target_sr)
        
        # 3. Silence Trimming
        # top_db=20 is a common default, adjusting as needed. Prompt didn't specify db.
        y, _ = librosa.effects.trim(y)
        
        # 4. Normalization to -1..1
        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val
            
        return y, target_sr

    def _chunk_audio(self, y, sr, chunk_duration=30):
        """
        If audio is longer than 30 seconds, split into chunks.
        """
        duration = len(y) / sr
        chunks = []
        if duration > chunk_duration:
            samples_per_chunk = int(chunk_duration * sr)
            total_samples = len(y)
            for start in range(0, total_samples, samples_per_chunk):
                end = min(start + samples_per_chunk, total_samples)
                chunks.append(y[start:end])
        else:
            chunks.append(y)
        return chunks

    def _calculate_smoothness(self, embeddings: torch.Tensor) -> float:
        """
        Calculates temporal smoothness.
        AI voices tend to have higher frame-to-frame cosine similarity (less 'jitter').
        """
        if embeddings.shape[1] < 2:
            return 0.0
            
        # Compare all frames with their next frame
        similarity = self.cos_sim(embeddings[0, :-1, :], embeddings[0, 1:, :])
        return float(similarity.mean().item())

    def _calculate_pitch_score(self, y, sr):
        """
        Estimates 'Human-ness' based on Pitch (F0) variance and jitter.
        Real voices have higher pitch standard deviation and frame-to-frame jitter.
        Returns score 0.0 (Robotic) to 1.0 (Very Human).
        """
        try:
            # Estimate pitch using pyin (Probabilistic YIN) - Robust to noise
            # fmin=50Hz (Deep male), fmax=1000Hz (High female/Screams)
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=1000, sr=sr)
            
            # Filter unvoiced
            f0 = f0[~np.isnan(f0)]
            
            if len(f0) < 10:
                print("DEBUG: Pitch Analysis -> Too few voiced frames.")
                return 0.0, 0.0, 0.0
                
            # 1. Pitch Standard Deviation (Intonation richness)
            pitch_std = np.std(f0)
            
            # 2. Jitter Proxy (Frame-to-frame absolute difference)
            jitter = np.mean(np.abs(np.diff(f0)))
            
            # Normalize (Heuristics) - "Goldilocks Trapezoid" based on pYIN
            # Human Jitter: 2Hz - 8Hz is the "Sweet Spot".
            # < 1Hz: Robotic.
            # > 10Hz: Unnatural/Noisy.
            
            if jitter < 1.0:
                score_jitter = 0.0
            elif 1.0 <= jitter < 2.0:
                score_jitter = (jitter - 1.0) # Ramp 0->1
            elif 2.0 <= jitter <= 8.0:
                score_jitter = 1.0 # Sweet spot
            elif 8.0 < jitter < 12.0:
                score_jitter = 1.0 - ((jitter - 8.0) / 4.0) # Ramp 1->0
            else: # > 12.0
                score_jitter = 0.0
                 
            # Std Score
            if pitch_std < 5.0:
                score_std = 0.0 # Monotone
            else:
                score_std = min(1.0, pitch_std / 20.0) # 25Hz std is good
            
            # Weight Jitter higher (80%) because intonation (std) is easy to fake
            final_score = (score_std * 0.2) + (score_jitter * 0.8)
            
            print(f"DEBUG: Pitch Analysis -> Std={pitch_std:.2f} (Score={score_std:.2f}), Jitter={jitter:.2f} (Score={score_jitter:.2f}) -> Final={final_score:.2f}")
            
            return final_score, pitch_std, jitter
        except Exception as e:
            print(f"Pitch calculation error: {e}")
            return 0.0, 0.0, 0.0
        except Exception as e:
            print(f"Pitch calculation error: {e}")
            return 0.0

    def detect_fraud(self, input_audio):
        # Initialize diagnostics
        smoothness = 0.0
        time_variance = 0.0
        heuristic_score = 0.0
        probs = None
        pitch_score = 0.0
        
        # --- Audio Loading & Preprocessing ---
        # --- Audio Loading & Preprocessing ---
        raw_y, raw_sr = self._load_audio(input_audio)
        y, sr = self._preprocess_audio(raw_y, raw_sr)
        # --- Primary AI vs Human detection ---
        # SUPER OPTIMIZATION: Hard cap to 3 seconds.
        # 16000 Hz * 3 seconds = 48000 samples
        max_samples = 16000 * 3
        if len(y) > max_samples:
            y = y[:max_samples]
            
        # Re-chunking is trivial now (it will be 1 chunk)
        chunks = self._chunk_audio(y, sr)  
        
        ai_probs = []
        
        for chunk in chunks:
            # Prepare inputs
            # Wav2Vec2 inputs
            # Processor requires list of numpy arrays, but we usually pass one by one or batched.
            # padding=True/False depends on if we batch. Here iterative.
            inputs = self.feature_extractor(
                chunk, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Apply softmax
            probs = F.softmax(logits, dim=-1)
            print(f"DEBUG: Probs: {probs[0].tolist()}, Labels: {self.model.config.id2label}")
            
            # Check model labels
            # Confirmed via debug: {0: 'real', 1: 'fake'}
            # Index 1 is AI/Fake.
            p_ai_chunk = probs[0][1].item()
            
            # Robustness check: if model has id2label, we could use it.
            ai_probs.append(p_ai_chunk)
            
        # Aggregate
        p_ai_model = sum(ai_probs) / len(ai_probs) if ai_probs else 0.0
        
        # --- Heuristic Analysis ---
        # Get embeddings from the last chunk for heuristic analysis (or average if feasible, but last is simpler)
        # We need to run the feature extractor again if we didn't save embeddings, 
        # BUT we can just run it on the full audio or a representative chunk.
        # Let's run on the first chunk for efficiency.
        
        heuristic_score = 0.0
        print(f"DEBUG: Num Chunks: {len(chunks)}")
        if len(chunks) > 0:
            print(f"DEBUG: Entering heuristic block. Num chunks: {len(chunks)}")
            # Re-run to get hidden states for smoothness/variance
            chk = chunks[0]
            with torch.no_grad():
                inp = self.feature_extractor(chk, sampling_rate=sr, return_tensors="pt", padding=True)
                out = self.model(**inp, output_hidden_states=True)
                # Wav2Vec2 hidden states are a tuple, taking the last one
                embeddings = out.hidden_states[-1] 
            
            # 1. Variance (Richness)
            # Real voices have high variance. AI is flatter.
            np_embeds = embeddings[0].cpu().numpy()
            time_variance = np.var(np_embeds, axis=0).mean()
            
            # 2. Smoothness (Robotic consistency)
            smoothness = self._calculate_smoothness(embeddings)
            
            # Normalization (Approximate based on XLS-R behavior)
            # Smoothness > 0.92 is often AI
            # Variance < 0.01 is often AI
            
            # Heuristic Score Calculation
            # Raised smoothness threshold back to 0.92 to avoid false positives on short clips
            score_smooth = max(0, (smoothness - 0.92) * 10) 
            score_var = max(0, 1.0 - (time_variance * 50))
            
            heuristic_score = (score_smooth + score_var) / 2.0
            heuristic_score = np.clip(heuristic_score, 0.0, 1.0)
            
        # --- Hybrid Fusion ---
        # 1. Pitch "Human Rescue" (Physics Check)
        pitch_score, p_std, p_jitter = self._calculate_pitch_score(y, sr)
        
        print(f"DEBUG: Pitch Human Score: {pitch_score:.3f}, Heuristic AI Score: {heuristic_score:.3f}, Model AI Prob: {p_ai_model:.3f}")
        
        # Base Model Probability
        final_p_ai = p_ai_model
        
        # 2. Heuristic Adjustments (Boost AI score)
        if heuristic_score > 0.7: # Raised back to 0.7 to avoid over-triggering
             # Strong signal that audio is "Robotic/Smooth" -> Boost AI score
             final_p_ai = max(final_p_ai, heuristic_score)
             
        # 3. Human Rescue
        # Removed strict heuristic block. Short clips might seem smooth but have human pitch.
        if pitch_score > 0.75 and final_p_ai < 0.98: # Allow rescue even if model is 0.95 (up to 0.98 now)
            # Cap AI probability if it's high
            if final_p_ai > 0.5:
                # Strong Human Features detected.
                reduction_factor = pitch_score * 0.5 # Increased reduction slightly (0.4 -> 0.5)
                final_p_ai = max(0.1, final_p_ai - reduction_factor)
                print(f"DEBUG: Human Rescue Triggered -> Pitch={pitch_score}, New Prob={final_p_ai}")
                print(f"DEBUG: Before Return -> Pitch={pitch_score}, Smooth={smoothness}, Var={time_variance}, Probs={probs}")
        
        classification = "AI" if final_p_ai > 0.5 else "Human"
        confidence = max(final_p_ai, 1 - final_p_ai)
        p_ai = final_p_ai # Update for reporting
        
        # --- Transcription and language detection (DISABLED) ---
        transcription = "Fraud detection disabled for hackathon optimization"
        english_translation = "Fraud detection disabled"
        detected_language = "N/A"
        
        # --- Fraud Keyword Analysis (DISABLED) ---
        found_keywords = []
        overall_risk = "LOW"
        
        # --- Explanation String ---
        parts = []
        parts.append(f"AI probability {round(p_ai, 2)}")
        parts.append(f"Deepfake detector classified as {classification}")
        if heuristic_score > 0.5:
             parts.append("Robotic voice patterns detected")
        elif pitch_score > 0.75:
             parts.append("Natural human pitch variations detected")
        
        explanation = ", ".join(parts)
        
        # Calculate audio duration for diagnostics
        audio_duration_seconds = round(len(y) / sr, 2)
        
        return {
            "classification": classification,
            "confidence_score": round(confidence, 2), # "confidence = max(p_ai, 1 - p_ai)"
            "ai_probability": round(p_ai, 2),
            "detected_language": detected_language,
            "transcription": transcription,
            "english_translation": english_translation,
            "fraud_keywords": found_keywords,
            "overall_risk": overall_risk,
            "explanation": explanation,
            # Diagnostic info
            "audio_duration_seconds": audio_duration_seconds,
            "num_chunks_processed": len(chunks),
            "chunk_ai_probabilities": [round(p, 3) for p in ai_probs],
            # Deep diagnostics
            "heuristic_score": round(heuristic_score, 3),
            "pitch_human_score": round(pitch_score, 3),
            "pitch_std": round(p_std, 2),
            "pitch_jitter": round(p_jitter, 2),
            "smoothness_score": round(smoothness, 4),
            "variance_score": round(time_variance, 5),
            "debug_probs": [round(p, 4) for p in probs[0].tolist()] if probs is not None else [],
            "debug_labels": self.model.config.id2label if self.model.config.id2label else "None"
        }

# Global instance
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = VoiceDetector.get_instance()
    return detector
