"""
Simple AI Voice Detector using pre-trained deepfake detection model
Uses MelodyMachine/Deepfake-audio-detection-V2 for reliable detection
"""
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
from fastapi import HTTPException
import io
import requests

class SimpleVoiceDetector:
    _instance = None

    def __init__(self):
        print("Initializing Simple Detector with pre-trained model...")

        # Use a pre-trained deepfake detection model
        self.model_name = "MelodyMachine/Deepfake-audio-detection-V2"

        try:
            print(f"Loading model: {self.model_name}")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.eval()
            print("✓ Pre-trained deepfake detection model loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load pre-trained model: {e}")
            print("Falling back to wav2vec2-base...")
            # Fallback to base model
            self.model_name = "facebook/wav2vec2-base"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(
                "facebook/wav2vec2-base",
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            self.model.eval()
            print("✓ Fallback model loaded")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_audio(self, input_audio):
        """Load audio from various input formats"""
        # If input is URL
        if isinstance(input_audio, str) and input_audio.startswith("http"):
            response = requests.get(input_audio)
            response.raise_for_status()
            audio_bytes = io.BytesIO(response.content)
        # If input is bytes-like
        elif isinstance(input_audio, (bytes, bytearray, io.BytesIO)):
            if isinstance(input_audio, (bytes, bytearray)):
                audio_bytes = io.BytesIO(input_audio)
            else:
                audio_bytes = input_audio
        elif isinstance(input_audio, np.ndarray):
            return input_audio, 16000
        else:
            audio_bytes = input_audio

        # Load with Librosa
        try:
            y, sr = librosa.load(audio_bytes, sr=None)
            return y, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")

    def _preprocess_audio(self, y, sr):
        """Simple preprocessing: convert to 16kHz mono"""
        target_sr = 16000

        # Resample if needed
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # Ensure mono
        if y.ndim > 1:
            y = librosa.to_mono(y)

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)

        # Normalize
        if np.abs(y).max() > 0:
            y = y / np.abs(y).max()

        return y, target_sr

    def detect_fraud(self, input_audio, metadata=None):
        """
        Simplified detection using pre-trained model
        """
        import time
        start_time = time.time()

        # Load and preprocess audio
        raw_y, raw_sr = self._load_audio(input_audio)
        y, sr = self._preprocess_audio(raw_y, raw_sr)

        if y is None or y.size == 0:
            raise HTTPException(status_code=400, detail="Audio processing failed")

        # Limit to 3 seconds for speed
        max_samples = 16000 * 3
        if len(y) > max_samples:
            y = y[:max_samples]

        # Run model inference
        try:
            inputs = self.feature_extractor(
                y,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                max_length=16000 * 10  # 10 seconds max
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)

            # Get prediction
            # Check if model has id2label mapping
            if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                labels = self.model.config.id2label
                print(f"DEBUG: Model labels: {labels}")
                print(f"DEBUG: Probabilities: {probs[0].tolist()}")

                # Find which label corresponds to fake/AI
                fake_idx = None
                real_idx = None

                for idx, label in labels.items():
                    label_lower = str(label).lower()
                    if 'fake' in label_lower or 'generated' in label_lower or 'spoof' in label_lower:
                        fake_idx = idx
                    elif 'real' in label_lower or 'bonafide' in label_lower or 'genuine' in label_lower:
                        real_idx = idx

                if fake_idx is not None:
                    p_ai = probs[0][fake_idx].item()
                elif real_idx is not None:
                    p_ai = 1.0 - probs[0][real_idx].item()
                else:
                    # Assume label 1 is fake (common convention)
                    p_ai = probs[0][1].item() if probs.shape[1] > 1 else probs[0][0].item()
            else:
                # No labels - assume index 1 is fake
                p_ai = probs[0][1].item() if probs.shape[1] > 1 else 0.5

            print(f"DEBUG: Raw model AI probability: {p_ai:.3f}")

            # HYBRID APPROACH: Aggressive audio quality + spectral checks
            try:
                # 1. SNR calculation (signal-to-noise ratio)
                noise_floor = np.percentile(np.abs(y), 10)
                signal_peak = np.percentile(np.abs(y), 90)

                if noise_floor > 0:
                    snr_db = 20 * np.log10(signal_peak / noise_floor)
                else:
                    snr_db = 60  # Very clean

                # 2. Energy variance (AI voices are more consistent)
                frame_length = 2048
                hop_length = 512
                rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                energy_variance = np.std(rms_frames) / (np.mean(rms_frames) + 1e-10)

                # 3. Spectral flatness (AI voices are less noisy/more tonal)
                spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

                print(f"DEBUG: SNR={snr_db:.1f}dB, EnergyVar={energy_variance:.3f}, SpectralFlat={spectral_flatness:.3f}")

                # AGGRESSIVE OVERRIDE LOGIC
                p_ai_adjusted = p_ai

                # Strong AI indicators (override to AI)
                if snr_db > 50:  # Very clean
                    if p_ai < 0.5:  # Model is uncertain or says human
                        p_ai_adjusted = 0.75
                        print(f"DEBUG: OVERRIDE → AI (very clean: {snr_db:.1f}dB)")
                    else:
                        p_ai_adjusted = max(p_ai, 0.7)  # Boost if already leaning AI

                elif snr_db > 42 and energy_variance < 0.15:  # Clean + consistent energy
                    if p_ai < 0.5:
                        p_ai_adjusted = 0.65
                        print(f"DEBUG: OVERRIDE → AI (clean + consistent)")

                # Strong Human indicators (override to human)
                elif snr_db < 30:  # Noisy
                    if p_ai > 0.5:  # Model says AI
                        p_ai_adjusted = 0.35
                        print(f"DEBUG: OVERRIDE → HUMAN (noisy: {snr_db:.1f}dB)")
                    else:
                        p_ai_adjusted = min(p_ai, 0.3)  # Reinforce if already leaning human

                elif snr_db < 35 and energy_variance > 0.20:  # Moderate noise + variable energy
                    if p_ai > 0.5:
                        p_ai_adjusted = 0.40
                        print(f"DEBUG: OVERRIDE → HUMAN (noisy + variable)")

                # Spectral flatness check (AI is more tonal, less flat)
                elif spectral_flatness < 0.05 and snr_db > 40:  # Very tonal + clean = AI
                    if p_ai < 0.5:
                        p_ai_adjusted = 0.70
                        print(f"DEBUG: OVERRIDE → AI (very tonal)")

            except Exception as e:
                print(f"Hybrid check error: {e}")
                import traceback
                traceback.print_exc()
                p_ai_adjusted = p_ai

            print(f"DEBUG: Final AI probability: {p_ai_adjusted:.3f}")

            # Classification
            classification = "AI" if p_ai_adjusted >= 0.5 else "Human"

            # Confidence calculation
            confidence = abs(p_ai_adjusted - 0.5) * 2.0
            confidence = max(0.5, min(1.0, confidence))

            # Use adjusted probability for reporting
            p_ai = p_ai_adjusted

            end_time = time.time()
            print(f"[TIMING] Detection time: {(end_time - start_time)*1000:.2f} ms")

            return {
                "classification": classification,
                "confidence_score": round(confidence, 2),
                "ai_probability": round(p_ai, 2),
                "detected_language": "N/A",
                "transcription": "Disabled for speed",
                "english_translation": "Disabled",
                "fraud_keywords": [],
                "overall_risk": "HIGH" if p_ai > 0.7 else "MEDIUM" if p_ai > 0.4 else "LOW",
                "explanation": f"AI probability: {p_ai:.2f}, Model: {self.model_name}",
                "audio_duration_seconds": round(len(y) / sr, 2),
                "num_chunks_processed": 1,
                "chunk_ai_probabilities": [round(p_ai, 3)],
                "heuristic_score": 0.0,
                "pitch_human_score": 0.0,
                "pitch_std": 0.0,
                "pitch_jitter": 0.0,
                "smoothness_score": 0.0,
                "variance_score": 0.0,
                "snr_score": 0.0,
                "debug_probs": [round(p, 4) for p in probs[0].tolist()],
                "debug_labels": self.model.config.id2label if hasattr(self.model.config, 'id2label') else {}
            }

        except Exception as e:
            import traceback
            print(f"Detection error: {e}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# Global instance
detector = None

def get_simple_detector():
    global detector
    if detector is None:
        detector = SimpleVoiceDetector.get_instance()
    return detector
