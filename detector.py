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
from fastapi import HTTPException

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

    def _calculate_snr(self, y: np.ndarray) -> float:
        """
        Calculates Signal-to-Noise Ratio (SNR) of the audio.
        High SNR (> 60dB) -> Studio quality (likely AI or studio rec).
        Lower SNR (< 30dB) -> Natural background noise (likely Human).
        """
        # Simple energy-based estimation
        # Assume lowest 10% energy frames are "noise" floor
        if len(y) < 100:
            return 0.0
            
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max) # 0 dB is max
        
        # Sort frame energies
        sorted_db = np.sort(rms_db[0])
        
        # Estimate noise floor (average of lowest 10% of frames)
        # Avoid silence trimming artifacts by taking 5th to 15th percentile
        noise_idx = int(len(sorted_db) * 0.1)
        if noise_idx == 0: noise_idx = 1
        noise_floor_db = np.mean(sorted_db[:noise_idx])
        
        # Signal power (average of top 20% of frames)
        signal_idx = int(len(sorted_db) * 0.8)
        signal_power_db = np.mean(sorted_db[signal_idx:])
        
        snr_value = signal_power_db - noise_floor_db
        return float(snr_value)

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

    def detect_fraud(self, input_audio, metadata=None):
        # Initialize diagnostics
        smoothness = 0.0
        time_variance = 0.0
        heuristic_score = 0.0
        probs = None
        pitch_score = 0.0
        snr_score = 0.0
        metadata_hit = False
        metadata_explanation = ""
        metadata_note = None
        
        # --- Metadata Short-Circuit (Instant Speed + High Accuracy) ---
        if metadata:
            encoder = metadata.get("encoder", "").lower()
            handler = metadata.get("handler_name", "").lower()
            
            # "Lavf" = Libavformat (FFmpeg). Almost all API-generated audio uses this.
            # "LAME" = Encoder often used in programmatic generation.
            # Real recordings usually have "iTunes", "Android", or no encoder tag.
            # Real recordings usually have "iTunes", "Android", or no encoder tag.
            if "lavf" in encoder or "lavc" in encoder or "google" in encoder:
                print(f"DEBUG: METADATA HIT! Encoder={encoder}. Marking as AI but continuing analysis.")
                metadata_hit = True
                metadata_explanation = f"Metadata analysis detected programmatic encoder: {metadata.get('encoder')}"

        # --- Audio Loading & Preprocessing ---
        raw_y, raw_sr = self._load_audio(input_audio)
        y, sr = self._preprocess_audio(raw_y, raw_sr)
        if y is None or y.size == 0:
            raise HTTPException(status_code=400, detail="Decoded audio contained no samples after preprocessing")
        
        # --- Primary AI vs Human detection ---
        # SUPER OPTIMIZATION: Increasing to 6 seconds for better context.
        # 16000 Hz * 6 seconds = 96000 samples
        max_samples = 16000 * 6
        if len(y) > max_samples:
            y = y[:max_samples]
            
        # Re-chunking is trivial now (it will be 1 chunk)
        chunks = [c for c in self._chunk_audio(y, sr) if len(c) > 0]
        if not chunks:
            raise HTTPException(status_code=400, detail="Audio contained no decodable frames")
        
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
        
        # 2. SNR/Noise Analysis
        snr_val = self._calculate_snr(y)
        # SNR > 50dB is very clean (Suspicious/AI-like)
        # SNR < 20dB is noisy (Likely Human)
        if snr_val > 50:
            snr_score = 1.0 # AI marker
        elif snr_val < 25:
            snr_score = -1.0 # Human marker (Negative favors human)
        else:
            snr_score = 0.0
            
        print(f"DEBUG: Physics -> PitchScore={pitch_score:.3f}, HeuristicAI={heuristic_score:.3f}, SNR={snr_val:.1f}, ModelProb={p_ai_model:.3f}")
        
        # Base Model Probability
        final_p_ai = p_ai_model
        
        # --- Heuristic Adjustments ---
        
        # A. Boost AI if Robotic (Smoothness + Variance + High SNR)
        # If Heuristic Score is high AND audio is super clean -> Boost AI
        if heuristic_score > 0.65: 
             boost = heuristic_score
             if snr_score > 0: # It's also super clean
                 boost = min(1.0, boost + 0.1)
             
             final_p_ai = max(final_p_ai, boost)

        # B. Human Rescue (Pitch Physics + Background Noise)
        # CRITICAL: Rescue ONLY if Model ALSO agrees (< 50% AI).
        # High-quality AI (ElevenLabs) can fake pitch jitter, so we must respect the model.
        
        is_noisy_human = (snr_val < 25) # Natural background noise
        has_human_pitch = (pitch_score > 0.70) # Lowered from 0.75
        model_says_human = (p_ai_model < 0.50) # Model must also lean Human
        
        # Rescue Conditions:
        # 1. Model says Human (<50%) AND Strong Pitch Evidence
        # 2. Model says Human (<50%) AND Moderate Pitch + Background Noise
        
        should_rescue = False
        rescue_strength = 0.0
        
        if model_says_human and has_human_pitch:
            should_rescue = True
            rescue_strength = pitch_score * 0.6 # Stronger rescue (was 0.5)
            
        if model_says_human and is_noisy_human and pitch_score > 0.4:
            # If it's noisy and has even mediocre pitch variance, it's likely human
            # (AI usually doesn't simulate background noise + pitch jitter together well)
            should_rescue = True
            rescue_strength = max(rescue_strength, 0.4)
            
        if should_rescue and final_p_ai < 0.99: # Allow rescue even for high confidence
            original_p = final_p_ai
            final_p_ai = max(0.05, final_p_ai - rescue_strength)
            print(f"DEBUG: Human Rescue Triggered -> Pitch={pitch_score}, SNR={snr_val}, RescueStrength={rescue_strength:.2f}, {original_p:.2f}->{final_p_ai:.2f}")

        # C. Metadata Note (no longer overrides classification)
        # Metadata is treated as a weak prior only; audio evidence must exist.
        if metadata_hit:
            metadata_note = metadata_explanation or "Suspicious encoder metadata detected"
            final_p_ai = min(1.0, final_p_ai + 0.1)
        
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
        if metadata_note:
             parts.append(metadata_note)
        if heuristic_score > 0.5:
             parts.append("Robotic voice patterns detected")
        elif pitch_score > 0.75:
             parts.append("Natural human pitch variations detected")
        if snr_score < 0:
             parts.append("Natural background noise detected")
        elif snr_score > 0:
             parts.append("Studio-quality silence detected")
        
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
            "snr_score": round(snr_val, 2) if 'snr_val' in locals() else 0.0,
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
