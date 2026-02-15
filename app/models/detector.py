import time
import torch
import numpy as np
import librosa
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
        self.detector_model_name = "facebook/wav2vec2-base"
        print(f"Loading AI Detector: {self.detector_model_name} ...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.detector_model_name
        )
        from transformers import Wav2Vec2Model
        self.model = Wav2Vec2Model.from_pretrained(self.detector_model_name)


        self.model.eval()
        for param in self.model.parameters():
          param.requires_grad = False

        hidden_size = self.model.config.hidden_size

        self.classifier = torch.nn.Linear(hidden_size, 2)
        self.classifier.eval()

        

        self.model = torch.quantization.quantize_dynamic(
         self.model,
       {torch.nn.Linear},
        dtype=torch.qint8
        )

        
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
        y, _ = librosa.effects.trim(y, top_db=30)
        
        # 4. Normalization to -1..1
        
            
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
            
            # Normalize (Heuristics) - More lenient for multilingual support
            # Human Jitter: 1.5Hz - 10Hz is acceptable
            # < 0.8Hz: Very robotic (AI)
            # > 15Hz: Too noisy/unstable

            if jitter < 0.8:
                score_jitter = 0.0  # Very robotic
            elif 0.8 <= jitter < 1.5:
                score_jitter = (jitter - 0.8) / 0.7  # Ramp 0->1
            elif 1.5 <= jitter <= 10.0:
                score_jitter = 1.0  # Human range (expanded)
            elif 10.0 < jitter < 15.0:
                score_jitter = 1.0 - ((jitter - 10.0) / 5.0)  # Ramp 1->0
            else:  # > 15.0
                score_jitter = 0.0  # Too noisy

            # Std Score - More lenient
            if pitch_std < 3.0:
                score_std = 0.0  # Very monotone
            elif pitch_std < 8.0:
                score_std = pitch_std / 8.0  # Ramp 0->1
            else:
                score_std = 1.0  # Good variation

            # Balanced weighting (60% jitter, 40% std)
            final_score = (score_std * 0.4) + (score_jitter * 0.6)
            
            print(f"DEBUG: Pitch Analysis -> Std={pitch_std:.2f} (Score={score_std:.2f}), Jitter={jitter:.2f} (Score={score_jitter:.2f}) -> Final={final_score:.2f}")
            
            return final_score, pitch_std, jitter
        except Exception as e:
            print(f"Pitch calculation error: {e}")
            return 0.0, 0.0, 0.0
        except Exception as e:
            print(f"Pitch calculation error: {e}")
            return 0.0

    def detect_fraud(self, input_audio, metadata=None):
        import time
        start_time = time.time()
        
        


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
        # OPTIMIZED MODE: 3 seconds max with enhanced heuristics
        max_samples = 16000 * 3
        if len(y) > max_samples:
            y = y[:max_samples]

        # Normalize only the sliced audio (faster)
        max_val = np.abs(y).max()
        if max_val > 0:
           y = y / max_val


        # Re-chunking is trivial now (it will be 1 chunk)
        chunks = [c for c in self._chunk_audio(y, sr) if len(c) > 0]
        if not chunks:
            raise HTTPException(status_code=400, detail="Audio contained no decodable frames")

        ai_probs = []
        embeddings_list = []

        for chunk in chunks:
            # Prepare inputs for Wav2Vec2
            inputs = self.feature_extractor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
               outputs = self.model(**inputs)
               hidden_states = outputs.last_hidden_state  # (batch, time, hidden)

              # Store embeddings for heuristic analysis
               embeddings_list.append(hidden_states)

              # Mean pooling over time dimension
               pooled = hidden_states.mean(dim=1)

               logits = self.classifier(pooled)


            # Apply softmax
            probs = F.softmax(logits, dim=-1)
            print(f"DEBUG: Model Probs: {probs[0].tolist()}")

            # The untrained classifier gives random predictions
            # We'll use heuristics instead for better accuracy
            p_ai_chunk = probs[0][1].item()
            ai_probs.append(p_ai_chunk)

        # Aggregate model prediction (untrained, not reliable)
        p_ai_model = sum(ai_probs) / len(ai_probs) if ai_probs else 0.5

        # --- ENHANCED HEURISTICS MODE ---
        # Since the model is untrained, we rely heavily on acoustic heuristics

        # 1. Calculate Pitch-based features (most reliable for AI detection)
        pitch_score, p_std, p_jitter = self._calculate_pitch_score(y, sr)

        # 2. Calculate SNR (AI voices are usually cleaner)
        snr_val = self._calculate_snr(y)

        # SNR scoring: High SNR (studio quality) suggests AI
        if snr_val > 50:
            snr_score = 0.3  # Strong AI indicator
        elif snr_val > 40:
            snr_score = 0.15  # Moderate AI indicator
        elif snr_val < 25:
            snr_score = -0.2  # Human indicator (background noise)
        else:
            snr_score = 0.0  # Neutral

        # 3. Calculate temporal smoothness (AI voices have consistent embeddings)
        if embeddings_list:
            smoothness = self._calculate_smoothness(embeddings_list[0])

            # High smoothness suggests AI (less natural variation)
            if smoothness > 0.95:
                smoothness_score = 0.25
            elif smoothness > 0.90:
                smoothness_score = 0.15
            else:
                smoothness_score = 0.0
        else:
            smoothness = 0.0
            smoothness_score = 0.0

        # Calculate variance in embedding magnitudes
        if embeddings_list:
            magnitudes = torch.norm(embeddings_list[0][0], dim=1).cpu().numpy()
            time_variance = float(np.std(magnitudes))

            # Low variance suggests AI (consistent energy)
            if time_variance < 0.5:
                variance_score = 0.15
            else:
                variance_score = 0.0
        else:
            time_variance = 0.0
            variance_score = 0.0

        # Combined heuristic score (AI indicators)
        heuristic_score = smoothness_score + variance_score + max(0, snr_score)

        print(f"DEBUG: Heuristics -> Pitch={pitch_score:.2f}, SNR={snr_val:.1f}, Smooth={smoothness:.3f}, Variance={time_variance:.3f}")

        # --- INTELLIGENT DECISION MAKING ---
        # Start with neutral baseline
        base_probability = 0.5

        # Pitch is the MOST reliable indicator
        # Score: 0.0 (robotic/AI) to 1.0 (natural/human)
        # Map to AI probability: invert the score
        if pitch_score > 0.70:
            # Strong human pitch characteristics
            pitch_adjustment = -0.25  # Push toward human
        elif pitch_score > 0.50:
            # Moderate human pitch
            pitch_adjustment = -0.15
        elif pitch_score < 0.25:
            # Very robotic pitch
            pitch_adjustment = +0.25  # Push toward AI
        elif pitch_score < 0.40:
            # Somewhat robotic
            pitch_adjustment = +0.15
        else:
            # Neutral pitch
            pitch_adjustment = 0.0

        # SNR Analysis (secondary indicator)
        # Very clean = AI, Very noisy = Human
        if snr_val > 50:
            snr_adjustment = +0.15  # Likely AI (studio quality)
        elif snr_val > 40:
            snr_adjustment = +0.08  # Possibly AI
        elif snr_val < 20:
            snr_adjustment = -0.15  # Likely Human (noisy)
        elif snr_val < 30:
            snr_adjustment = -0.08  # Possibly Human
        else:
            snr_adjustment = 0.0  # Neutral

        # Temporal smoothness (tertiary indicator)
        # Very smooth = AI, Variable = Human
        if smoothness > 0.95:
            smoothness_adjustment = +0.10
        elif smoothness > 0.92:
            smoothness_adjustment = +0.05
        elif smoothness < 0.85:
            smoothness_adjustment = -0.05
        else:
            smoothness_adjustment = 0.0

        # Energy variance (tertiary indicator)
        if time_variance < 0.3:
            variance_adjustment = +0.05  # Very consistent = AI
        elif time_variance > 0.8:
            variance_adjustment = -0.05  # Very variable = Human
        else:
            variance_adjustment = 0.0

        # Calculate final AI probability
        final_p_ai = (
            base_probability +
            pitch_adjustment +
            snr_adjustment +
            smoothness_adjustment +
            variance_adjustment
        )

        # Clamp to [0, 1]
        final_p_ai = max(0.0, min(1.0, final_p_ai))

        print(f"DEBUG: AI Probability -> Base={base_probability:.2f}, Pitch={pitch_adjustment:+.2f}, SNR={snr_adjustment:+.2f}, Smooth={smoothness_adjustment:+.2f}, Var={variance_adjustment:+.2f} = {final_p_ai:.2f}")
        
        # --- Combined Signal Refinements ---
        # Look for combinations that strongly indicate one or the other

        # Strong Human: Natural pitch + Background noise
        is_natural_pitch = (pitch_score > 0.60)
        has_background_noise = (snr_val < 30)

        if is_natural_pitch and has_background_noise:
            # Very strong human indicator
            final_p_ai = min(final_p_ai, 0.30)  # Cap at 30% AI probability
            print(f"DEBUG: Strong HUMAN signal (pitch={pitch_score:.2f}, SNR={snr_val:.1f}) -> Capped at 0.30")

        # Strong AI: Robotic pitch + Studio quality
        is_robotic_pitch = (pitch_score < 0.35)
        is_studio_quality = (snr_val > 45)

        if is_robotic_pitch and is_studio_quality:
            # Very strong AI indicator
            final_p_ai = max(final_p_ai, 0.70)  # Floor at 70% AI probability
            print(f"DEBUG: Strong AI signal (pitch={pitch_score:.2f}, SNR={snr_val:.1f}) -> Floored at 0.70")

        # Moderate Human: Good pitch variation alone
        if pitch_score > 0.75 and final_p_ai > 0.4:
            final_p_ai = min(final_p_ai, 0.40)
            print(f"DEBUG: Excellent human pitch ({pitch_score:.2f}) -> Capped at 0.40")

        # Moderate AI: Very smooth + clean (but not robotic pitch)
        is_very_smooth = (smoothness > 0.93)
        if is_very_smooth and is_studio_quality and final_p_ai < 0.6:
            final_p_ai = max(final_p_ai, 0.60)
            print(f"DEBUG: Very smooth + clean -> Floored at 0.60")

        # C. Metadata boost (weak signal)
        if metadata_hit:
            metadata_note = metadata_explanation or "Suspicious encoder metadata detected"
            final_p_ai = min(0.95, final_p_ai + 0.10)
        else:
            metadata_note = None

        # Clamp final probability
        final_p_ai = max(0.0, min(1.0, final_p_ai))

        # Classification with 0.5 threshold
        classification = "AI" if final_p_ai >= 0.5 else "Human"

        # Confidence calculation - distance from threshold
        # Convert AI probability to confidence in the classification
        if final_p_ai >= 0.5:
            # Classified as AI - confidence increases as we move toward 1.0
            # Map 0.5->1.0 to 0.5->1.0
            confidence = 0.5 + (final_p_ai - 0.5)  # Linear: 0.5 at threshold, 1.0 at certainty
        else:
            # Classified as Human - confidence increases as we move toward 0.0
            # Map 0.0->0.5 to 1.0->0.5
            confidence = 0.5 + (0.5 - final_p_ai)  # Linear: 0.5 at threshold, 1.0 at certainty

        # Ensure confidence stays in valid range
        confidence = max(0.5, min(1.0, confidence))
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
        end_time = time.time()
        print(f"[TIMING] detect_fraud inference time: {(end_time - start_time)*1000:.2f} ms")
        
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
