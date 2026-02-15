import time
import torch
import numpy as np
import librosa
import io
import requests
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch.nn.functional as F
from fastapi import HTTPException
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from torch.nn import CosineSimilarity

@dataclass
class WindowEmbedding:
    """Stores embedding for a temporal window"""
    start_frame: int
    end_frame: int
    embedding: torch.Tensor
    ai_probability: float
    
@dataclass
class SpeakerChange:
    """Detected speaker change event"""
    window_index: int
    similarity_score: float
    timestamp_seconds: float
    detection_method: str  # 'embedding' or 'acoustic'

@dataclass
class EscalationEvent:
    """Detected escalation in AI probability"""
    window_index: int
    probability_jump: float
    timestamp_seconds: float

class VoiceDetectorV2:
    """
    Multi-phase fraud detection system with:
    - Phase 1: XLS-R multilingual backbone
    - Phase 2: Window-level embeddings
    - Phase 3: Speaker change detection (dual-method)
    - Phase 4: Escalation detection
    - Phase 5: Short clip stabilization
    - Phase 6: Telecom audio handling
    - Phase 7: Mixed language robustness
    - Phase 8: Over-polite AI pattern detection
    - Phase 9: Legit automated call handling
    - Phase 10: Multi-speaker logic
    - Phase 11: Latency optimization
    - Phase 12: Trainable classifier head
    - Phase 13: ElevenLabs-specific detection (INTEGRATED)
    """
    
    _instance = None
    
    def __init__(self, config: Optional[Dict] = None):
        print("Initializing Enhanced Detection Pipeline V2 (with ElevenLabs detection)...")
        
        # Configuration with sensible defaults
        self.config = config or {}
        
        # ============= PHASE 1: XLS-R Multilingual Backbone =============
        self.detector_model_name = "facebook/wav2vec2-xls-r-300m"
        print(f"Loading XLS-R Model: {self.detector_model_name}")
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.detector_model_name
        )
        self.model = Wav2Vec2Model.from_pretrained(self.detector_model_name)
        
        # Freeze backbone (Phase 12: only classifier trains)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        hidden_size = self.model.config.hidden_size  # 1024 for XLS-R-300M
        
        # ============= PHASE 12: Trainable Classifier Head =============
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 2)
        )
        self.classifier.eval()  # Set to eval by default, train() when training
        
        # Quantization for inference speed (Phase 11)
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # ============= PHASE 2: Window Configuration =============
        self.window_frames = self.config.get('window_frames', 10)
        self.window_overlap = self.config.get('window_overlap', 5)
        
        # ============= PHASE 3: Speaker Change Detection =============
        self.speaker_change_threshold = self.config.get('speaker_change_threshold', 0.65)
        self.acoustic_change_threshold = self.config.get('acoustic_change_threshold', 0.30)
        self.use_acoustic_detection = self.config.get('use_acoustic_detection', True)
        self.cos_sim = CosineSimilarity(dim=1, eps=1e-6)
        
        # ============= PHASE 4: Escalation Detection =============
        self.escalation_variance_threshold = self.config.get('escalation_variance_threshold', 0.15)
        self.escalation_jump_threshold = self.config.get('escalation_jump_threshold', 0.25)
        
        # ============= PHASE 5: Short Clip Handling =============
        self.min_reliable_duration = self.config.get('min_reliable_duration', 1.5)
        
        # ============= PHASE 6: Telecom Audio Thresholds =============
        self.telecom_snr_threshold = self.config.get('telecom_snr_threshold', 15.0)
        
        # ============= PHASE 8: Over-Polite AI Detection =============
        self.flat_energy_threshold = self.config.get('flat_energy_threshold', 0.05)
        self.flat_pitch_threshold = self.config.get('flat_pitch_threshold', 1.0)
        
        # ============= PHASE 11: Latency Optimization =============
        self.max_audio_seconds = self.config.get('max_audio_seconds', 15.0)
        self.target_latency_ms = self.config.get('target_latency_ms', 2000)
        
        # ============= PHASE 13: ElevenLabs-Specific Detection =============
        self.use_elevenlabs_detection = self.config.get('use_elevenlabs_detection', True)
        
        # ElevenLabs thresholds (research-based)
        self.elevenlabs_prosody_threshold = self.config.get('elevenlabs_prosody_threshold', 0.15)
        self.elevenlabs_pause_threshold = self.config.get('elevenlabs_pause_threshold', 0.20)
        self.elevenlabs_spectral_threshold = self.config.get('elevenlabs_spectral_threshold', 0.01)
        self.elevenlabs_pitch_threshold = self.config.get('elevenlabs_pitch_threshold', 0.80)
        self.elevenlabs_disfluency_threshold = self.config.get('elevenlabs_disfluency_threshold', 0.01)
        self.elevenlabs_emotion_threshold = self.config.get('elevenlabs_emotion_threshold', 0.10)
        
        print("✓ Enhanced Detection Pipeline V2 loaded successfully")
        print(f"  - Backbone: {self.detector_model_name}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Max audio duration: {self.max_audio_seconds}s")
        print(f"  - Speaker change threshold: {self.speaker_change_threshold}")
        print(f"  - ElevenLabs detection: {'ENABLED' if self.use_elevenlabs_detection else 'DISABLED'}")

    @classmethod
    def get_instance(cls, config: Optional[Dict] = None):
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def _load_audio(self, input_audio) -> Tuple[np.ndarray, int]:
        """Load audio from URL, bytes, or file path - supports all formats"""
        if isinstance(input_audio, str) and input_audio.startswith("http"):
            response = requests.get(input_audio)
            response.raise_for_status()
            audio_bytes = io.BytesIO(response.content)
        elif isinstance(input_audio, (bytes, bytearray, io.BytesIO)):
            if isinstance(input_audio, (bytes, bytearray)):
                audio_bytes = io.BytesIO(input_audio)
            else:
                audio_bytes = input_audio
        elif isinstance(input_audio, np.ndarray):
            return input_audio, 16000
        else:
            audio_bytes = input_audio

        try:
            y, sr = librosa.load(audio_bytes, sr=None)
            return y, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")

    def _preprocess_audio(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Preprocess audio: mono, resample, trim, normalize"""
        target_sr = 16000
        
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        if y.ndim > 1:
            y = librosa.to_mono(y)
        
        y, _ = librosa.effects.trim(y, top_db=30)
        
        return y, target_sr

    def _smart_sample_long_audio(self, y: np.ndarray, sr: int, target_duration: float = 15.0) -> np.ndarray:
        """Smart sampling for very long audio - samples beginning, middle, end"""
        duration = len(y) / sr
        
        if duration <= target_duration:
            return y
        
        segment_duration = target_duration / 3.0
        segment_samples = int(sr * segment_duration)
        
        segments = []
        segments.append(y[:segment_samples])
        
        mid_start = int(len(y) / 2 - segment_samples / 2)
        segments.append(y[mid_start:mid_start + segment_samples])
        
        segments.append(y[-segment_samples:])
        
        return np.concatenate(segments)

    # ============= PHASE 2: Window-Level Embeddings =============
    
    def _extract_window_embeddings(self, hidden_states: torch.Tensor) -> List[WindowEmbedding]:
        """Extract embeddings for sliding windows"""
        batch_size, num_frames, hidden_dim = hidden_states.shape
        
        if num_frames < self.window_frames:
            pooled = hidden_states.mean(dim=1)
            logits = self.classifier(pooled)
            probs = F.softmax(logits, dim=-1)
            
            return [WindowEmbedding(
                start_frame=0,
                end_frame=num_frames,
                embedding=pooled[0],
                ai_probability=probs[0][1].item()
            )]
        
        windows = []
        step = self.window_frames - self.window_overlap
        
        for start_frame in range(0, num_frames - self.window_frames + 1, step):
            end_frame = start_frame + self.window_frames
            
            window_hidden = hidden_states[:, start_frame:end_frame, :]
            pooled = window_hidden.mean(dim=1)
            
            with torch.no_grad():
                logits = self.classifier(pooled)
                probs = F.softmax(logits, dim=-1)
            
            windows.append(WindowEmbedding(
                start_frame=start_frame,
                end_frame=end_frame,
                embedding=pooled[0],
                ai_probability=probs[0][1].item()
            ))
        
        return windows

    # ============= PHASE 3: Speaker Change Detection =============
    
    def _detect_acoustic_changes(self, y: np.ndarray, sr: int) -> Tuple[List[SpeakerChange], int]:
        """Acoustic-based speaker change detection"""
        segment_duration = 2.0
        segment_samples = int(sr * segment_duration)
        num_segments = len(y) // segment_samples
        
        if num_segments < 2:
            return [], 0
        
        changes = []
        prev_features = None
        
        for i in range(num_segments):
            start = i * segment_samples
            end = min(start + segment_samples, len(y))
            segment = y[start:end]
            
            energy = np.sqrt(np.mean(segment**2))
            zcr = np.sum(np.abs(np.diff(np.sign(segment)))) / (2 * len(segment))
            
            autocorr = np.correlate(segment, segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr[20:]
            if len(autocorr) > 0:
                peak_lag = np.argmax(autocorr) + 20
                pitch = sr / peak_lag if peak_lag > 0 else 0
            else:
                pitch = 0
            
            if prev_features is not None:
                energy_change = abs(energy - prev_features['energy']) / (prev_features['energy'] + 1e-6)
                zcr_change = abs(zcr - prev_features['zcr']) / (prev_features['zcr'] + 1e-6)
                pitch_change = abs(pitch - prev_features['pitch']) / (prev_features['pitch'] + 1.0)
                
                change_score = (energy_change + zcr_change + pitch_change) / 3.0
                
                if change_score > self.acoustic_change_threshold:
                    timestamp = i * segment_duration
                    changes.append(SpeakerChange(
                        window_index=i,
                        similarity_score=1.0 - change_score,
                        timestamp_seconds=timestamp,
                        detection_method='acoustic'
                    ))
            
            prev_features = {'energy': energy, 'zcr': zcr, 'pitch': pitch}
        
        return changes, len(changes)
    
    def _detect_speaker_changes(self, windows: List[WindowEmbedding], sample_rate: int) -> Tuple[List[SpeakerChange], int]:
        """Embedding-based speaker change detection"""
        if len(windows) < 2:
            return [], 0
        
        speaker_changes = []
        
        for i in range(len(windows) - 1):
            emb1 = windows[i].embedding.unsqueeze(0)
            emb2 = windows[i + 1].embedding.unsqueeze(0)
            
            similarity = self.cos_sim(emb1, emb2).item()
            
            if similarity < self.speaker_change_threshold:
                mid_frame = (windows[i].end_frame + windows[i + 1].start_frame) // 2
                timestamp = mid_frame / 50.0
                
                speaker_changes.append(SpeakerChange(
                    window_index=i,
                    similarity_score=similarity,
                    timestamp_seconds=timestamp,
                    detection_method='embedding'
                ))
        
        if len(windows) >= 4:
            first_emb = windows[0].embedding.unsqueeze(0)
            last_emb = windows[-1].embedding.unsqueeze(0)
            first_last_similarity = self.cos_sim(first_emb, last_emb).item()
            
            if first_last_similarity < 0.70 and len(speaker_changes) == 0:
                timestamp = windows[-1].start_frame / 50.0
                speaker_changes.append(SpeakerChange(
                    window_index=len(windows) - 1,
                    similarity_score=first_last_similarity,
                    timestamp_seconds=timestamp,
                    detection_method='embedding'
                ))
        
        return speaker_changes, len(speaker_changes)

    # ============= PHASE 4: Escalation Detection =============
    
    def _detect_escalation(self, windows: List[WindowEmbedding]) -> Tuple[List[EscalationEvent], float, bool]:
        """Detect escalation patterns in AI probability"""
        if len(windows) < 3:
            return [], 0.0, False
        
        ai_probs = [w.ai_probability for w in windows]
        variance = float(np.var(ai_probs))
        
        escalation_events = []
        
        for i in range(1, len(windows)):
            prob_jump = windows[i].ai_probability - windows[i-1].ai_probability
            
            if prob_jump > self.escalation_jump_threshold:
                timestamp = windows[i].start_frame / 50.0
                
                escalation_events.append(EscalationEvent(
                    window_index=i,
                    probability_jump=prob_jump,
                    timestamp_seconds=timestamp
                ))
        
        has_escalation = variance > self.escalation_variance_threshold or len(escalation_events) > 0
        
        return escalation_events, variance, has_escalation

    # ============= PHASE 6: Audio Quality Analysis =============
    
    def _analyze_audio_quality(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze audio quality for telecom detection"""
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        sorted_db = np.sort(rms_db[0])
        noise_idx = max(1, int(len(sorted_db) * 0.1))
        noise_floor_db = np.mean(sorted_db[:noise_idx])
        signal_idx = int(len(sorted_db) * 0.8)
        signal_power_db = np.mean(sorted_db[signal_idx:])
        
        snr = float(signal_power_db - noise_floor_db)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        avg_bandwidth = float(np.mean(spectral_bandwidth))
        
        is_telecom = snr < self.telecom_snr_threshold and avg_bandwidth < 4000
        
        return {
            'snr': snr,
            'bandwidth': avg_bandwidth,
            'is_telecom_quality': is_telecom
        }

    # ============= PHASE 8: Over-Polite AI Pattern Detection =============
    
    def _detect_overpolite_pattern(self, y: np.ndarray, sr: int) -> Dict:
        """Detect unnaturally flat energy and pitch patterns"""
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
        energy_variance = float(np.var(rms))
        
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=1000, sr=sr)
            f0_valid = f0[~np.isnan(f0)]
            
            if len(f0_valid) > 10:
                pitch_variance = float(np.var(f0_valid))
                pitch_jitter = float(np.mean(np.abs(np.diff(f0_valid))))
            else:
                pitch_variance = 0.0
                pitch_jitter = 0.0
        except:
            pitch_variance = 0.0
            pitch_jitter = 0.0
        
        is_flat_energy = energy_variance < self.flat_energy_threshold
        is_flat_pitch = pitch_jitter < self.flat_pitch_threshold
        
        is_overpolite = is_flat_energy and is_flat_pitch
        
        return {
            'energy_variance': energy_variance,
            'pitch_variance': pitch_variance,
            'pitch_jitter': pitch_jitter,
            'is_flat_energy': is_flat_energy,
            'is_flat_pitch': is_flat_pitch,
            'is_overpolite_ai': is_overpolite
        }

    # ============= PHASE 13: ElevenLabs-Specific Detection (INTEGRATED) =============
    
    def _detect_elevenlabs_features(self, y: np.ndarray, sr: int) -> Dict:
        """
        Integrated ElevenLabs detection using research-based acoustic features
        
        Based on 6 key differences between TTS and human speech:
        1. Prosody regularity
        2. Pause timing patterns
        3. Spectral cleanliness
        4. Pitch smoothness
        5. Disfluency absence
        6. Emotional flatness
        """
        if not self.use_elevenlabs_detection:
            return None
        
        features = {}
        
        # Feature 1: Prosody Regularity
        try:
            f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                             fmax=librosa.note_to_hz('C7'), sr=sr)
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 20:
                segment_size = len(f0_voiced) // 5
                variances = []
                for i in range(5):
                    start = i * segment_size
                    end = min(start + segment_size, len(f0_voiced))
                    if end - start > 1:
                        variances.append(np.var(f0_voiced[start:end]))
                
                if variances:
                    cv = np.var(variances) / (np.mean(variances) + 1e-6)
                    features['prosody_regular'] = cv < self.elevenlabs_prosody_threshold
                    features['prosody_score'] = 1.0 - min(cv / 0.5, 1.0)
                else:
                    features['prosody_regular'] = False
                    features['prosody_score'] = 0.0
            else:
                features['prosody_regular'] = False
                features['prosody_score'] = 0.0
        except:
            features['prosody_regular'] = False
            features['prosody_score'] = 0.0
        
        # Feature 2: Pause Regularity
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        silence_threshold = np.percentile(rms, 20)
        is_silent = rms < silence_threshold
        
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, silent in enumerate(is_silent):
            if silent and not in_pause:
                pause_start = i
                in_pause = True
            elif not silent and in_pause:
                pause_duration = (i - pause_start) * 512 / sr
                if pause_duration > 0.1:
                    pauses.append(pause_duration)
                in_pause = False
        
        if len(pauses) >= 3:
            pause_cv = np.std(pauses) / (np.mean(pauses) + 1e-6)
            features['pauses_regular'] = pause_cv < self.elevenlabs_pause_threshold
            features['pause_score'] = 1.0 - min(pause_cv / 0.5, 1.0)
        else:
            features['pauses_regular'] = False
            features['pause_score'] = 0.0
        
        # Feature 3: Spectral Cleanliness
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        centroid_cv = np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-6)
        bandwidth_cv = np.std(spectral_bandwidth) / (np.mean(spectral_bandwidth) + 1e-6)
        combined_variance = (centroid_cv + bandwidth_cv) / 2.0
        
        features['too_clean'] = combined_variance < self.elevenlabs_spectral_threshold
        features['spectral_score'] = 1.0 - min(combined_variance / 0.1, 1.0)
        
        # Feature 4: Pitch Smoothness
        try:
            f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=1000, sr=sr)
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 10:
                pitch_diff = np.abs(np.diff(f0_voiced))
                second_diff = np.abs(np.diff(pitch_diff))
                smoothness = 1.0 / (np.mean(second_diff) + 1e-6)
                normalized_smoothness = min(smoothness / 100.0, 1.0)
                
                features['pitch_smooth'] = normalized_smoothness > self.elevenlabs_pitch_threshold
                features['pitch_score'] = normalized_smoothness
            else:
                features['pitch_smooth'] = False
                features['pitch_score'] = 0.0
        except:
            features['pitch_smooth'] = False
            features['pitch_score'] = 0.0
        
        # Feature 5: Disfluency Detection
        rms_diff = np.abs(np.diff(rms))
        threshold = np.percentile(rms_diff, 90)
        sudden_changes = np.sum(rms_diff > threshold)
        disfluency_rate = sudden_changes / len(rms)
        
        features['no_disfluencies'] = disfluency_rate < self.elevenlabs_disfluency_threshold
        features['disfluency_score'] = 1.0 - min(disfluency_rate / 0.1, 1.0)
        
        # Feature 6: Emotional Flatness
        energy_cv = np.std(rms) / (np.mean(rms) + 1e-6)
        features['flat_emotion'] = energy_cv < self.elevenlabs_emotion_threshold
        features['emotion_score'] = 1.0 - min(energy_cv / 0.5, 1.0)
        
        # Aggregate ElevenLabs probability
        scores = [
            features['prosody_score'],
            features['pause_score'],
            features['spectral_score'],
            features['pitch_score'],
            features['disfluency_score'],
            features['emotion_score']
        ]
        
        # Weighted average (prosody and pitch most important)
        weights = [0.20, 0.15, 0.15, 0.20, 0.15, 0.15]
        elevenlabs_probability = sum(s * w for s, w in zip(scores, weights))
        
        # Count indicators
        tts_indicators = sum([
            features['prosody_regular'],
            features['pauses_regular'],
            features['too_clean'],
            features['pitch_smooth'],
            features['no_disfluencies'],
            features['flat_emotion']
        ])
        
        return {
            'elevenlabs_probability': float(elevenlabs_probability),
            'tts_indicators_count': tts_indicators,
            'is_likely_elevenlabs': elevenlabs_probability > 0.6,
            'features': features
        }

    # ============= PHASE 10: Multi-Speaker Risk Fusion =============
    
    def _calculate_final_risk(
        self,
        avg_ai_probability: float,
        speaker_changes: int,
        has_escalation: bool,
        duration_seconds: float,
        is_telecom: bool,
        is_overpolite: bool,
        metadata_suspicious: bool
    ) -> Tuple[str, str, float]:
        """Final risk calculation incorporating all signals"""
        final_ai_prob = avg_ai_probability
        
        # Phase 5: Short clip stabilization
        if duration_seconds < self.min_reliable_duration:
            final_ai_prob = min(0.65, final_ai_prob)
            if final_ai_prob > 0.4:
                return "AI", "MEDIUM", final_ai_prob
        
        # Phase 6: Telecom quality adjustment
        if is_telecom and avg_ai_probability < 0.7:
            final_ai_prob = max(0.0, final_ai_prob - 0.15)
        
        # Phase 8: Over-polite AI boost
        if is_overpolite and avg_ai_probability > 0.5:
            final_ai_prob = min(1.0, final_ai_prob + 0.15)
        
        # Phase 9: Legit automated system detection
        is_likely_ivr = (
            is_overpolite and 
            speaker_changes == 0 and 
            not has_escalation and
            not metadata_suspicious and
            duration_seconds < 15
        )
        
        if is_likely_ivr and final_ai_prob > 0.7:
            return "AUTOMATED_SYSTEM", "LOW", final_ai_prob
        
        # Phase 10: Multi-speaker logic
        if final_ai_prob > 0.65 and speaker_changes > 0 and has_escalation:
            return "AI", "HIGH", final_ai_prob
        
        if final_ai_prob > 0.80:
            return "AI", "HIGH", final_ai_prob
        
        if final_ai_prob > 0.50:
            risk = "MEDIUM" if speaker_changes == 0 else "HIGH"
            return "AI", risk, final_ai_prob
        
        return "Human", "LOW", final_ai_prob

    # ============= Main Detection Pipeline =============
    
    def detect_fraud(self, input_audio, metadata: Optional[Dict] = None) -> Dict:
        """
        Main fraud detection pipeline with all 13 phases integrated
        """
        start_time = time.time()
        
        # Metadata analysis
        metadata_suspicious = False
        metadata_note = None
        
        if metadata:
            encoder = metadata.get("encoder", "").lower()
            if "lavf" in encoder or "lavc" in encoder or "google" in encoder:
                metadata_suspicious = True
                metadata_note = f"Suspicious encoder: {metadata.get('encoder')}"
        
        # Load and preprocess audio
        raw_y, raw_sr = self._load_audio(input_audio)
        y, sr = self._preprocess_audio(raw_y, raw_sr)
        
        if y is None or y.size == 0:
            raise HTTPException(status_code=400, detail="Audio contained no samples")
        
        full_duration_seconds = len(y) / sr
        
        # Smart audio sampling
        if full_duration_seconds > 20.0:
            print(f"DEBUG: Long audio ({full_duration_seconds:.1f}s), using smart sampling")
            y = self._smart_sample_long_audio(y, sr, target_duration=15.0)
            duration_seconds = len(y) / sr
        else:
            max_samples = int(sr * self.max_audio_seconds)
            if len(y) > max_samples:
                y = y[:max_samples]
                duration_seconds = self.max_audio_seconds
            else:
                duration_seconds = full_duration_seconds
        
        # Normalize
        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val
        
        print(f"DEBUG: Processing {duration_seconds:.2f}s out of {full_duration_seconds:.2f}s total")
        
        # Extract features
        inputs = self.feature_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
        
        print(f"DEBUG: Generated {hidden_states.shape[1]} time frames")
        
        # Window-level embeddings
        windows = self._extract_window_embeddings(hidden_states)
        print(f"DEBUG: Created {len(windows)} windows")
        
        # Speaker change detection (dual method)
        embedding_changes_list, embedding_change_count = self._detect_speaker_changes(windows, sr)
        print(f"DEBUG: Embedding-based changes: {embedding_change_count}")
        
        acoustic_changes_list, acoustic_change_count = [], 0
        if self.use_acoustic_detection:
            acoustic_changes_list, acoustic_change_count = self._detect_acoustic_changes(y, sr)
            print(f"DEBUG: Acoustic-based changes: {acoustic_change_count}")
        
        if acoustic_change_count > embedding_change_count:
            speaker_changes_list = acoustic_changes_list
            speaker_change_count = acoustic_change_count
        else:
            speaker_changes_list = embedding_changes_list
            speaker_change_count = embedding_change_count
        
        # Escalation detection
        escalation_events, ai_prob_variance, has_escalation = self._detect_escalation(windows)
        
        # Audio quality analysis
        quality_metrics = self._analyze_audio_quality(y, sr)
        
        # Over-polite detection
        polite_metrics = self._detect_overpolite_pattern(y, sr)
        
        # ElevenLabs-specific detection
        elevenlabs_analysis = self._detect_elevenlabs_features(y, sr)
        elevenlabs_boost = 0.0
        
        if elevenlabs_analysis:
            if elevenlabs_analysis['elevenlabs_probability'] > 0.6:
                elevenlabs_boost = 0.20
                print(f"DEBUG: ElevenLabs detected! Prob: {elevenlabs_analysis['elevenlabs_probability']:.2%}")
            elif elevenlabs_analysis['elevenlabs_probability'] > 0.4:
                elevenlabs_boost = 0.10
        
        # Calculate average AI probability
        avg_ai_probability = np.mean([w.ai_probability for w in windows])
        avg_ai_probability = min(1.0, avg_ai_probability + elevenlabs_boost)
        
        # Final risk calculation
        classification, risk_level, final_ai_prob = self._calculate_final_risk(
            avg_ai_probability=avg_ai_probability,
            speaker_changes=speaker_change_count,
            has_escalation=has_escalation,
            duration_seconds=duration_seconds,
            is_telecom=quality_metrics['is_telecom_quality'],
            is_overpolite=polite_metrics['is_overpolite_ai'],
            metadata_suspicious=metadata_suspicious
        )
        
        # Build explanation
        explanation_parts = []
        explanation_parts.append(f"AI probability: {final_ai_prob:.2%}")
        explanation_parts.append(f"Classified as: {classification}")
        
        if speaker_change_count > 0:
            detection_methods = set(sc.detection_method for sc in speaker_changes_list)
            methods_str = "+".join(detection_methods)
            explanation_parts.append(f"Detected {speaker_change_count} speaker change(s) ({methods_str})")
        
        if has_escalation:
            explanation_parts.append("Escalation pattern detected")
        
        if quality_metrics['is_telecom_quality']:
            explanation_parts.append("Telecom-quality audio detected")
        
        if polite_metrics['is_overpolite_ai']:
            explanation_parts.append("Over-polite AI pattern detected")
        
        if elevenlabs_analysis and elevenlabs_analysis['is_likely_elevenlabs']:
            explanation_parts.append(f"ElevenLabs TTS detected ({elevenlabs_analysis['tts_indicators_count']}/6 indicators)")
        
        if metadata_suspicious:
            explanation_parts.append(metadata_note or "Suspicious metadata")
        
        if duration_seconds < self.min_reliable_duration:
            explanation_parts.append("Short clip - reduced confidence")
        
        if full_duration_seconds > 20.0:
            explanation_parts.append(f"Long audio ({full_duration_seconds:.1f}s) - sampled key segments")
        
        # Compile results
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        result = {
            # Core results
            "classification": str(classification),
            "risk_level": str(risk_level),
            "ai_probability": float(round(final_ai_prob, 3)),
            "confidence_score": float(round(max(final_ai_prob, 1 - final_ai_prob), 3)),
            
            # Multi-speaker analysis
            "speaker_changes": int(speaker_change_count),
            "speaker_change_events": [
                {
                    "timestamp": float(round(sc.timestamp_seconds, 2)),
                    "similarity": float(round(sc.similarity_score, 3)),
                    "method": str(sc.detection_method)
                }
                for sc in speaker_changes_list
            ],
            
            # Escalation analysis
            "has_escalation": bool(has_escalation),
            "ai_probability_variance": float(round(ai_prob_variance, 4)),
            "escalation_events": [
                {
                    "timestamp": float(round(ee.timestamp_seconds, 2)),
                    "probability_jump": float(round(ee.probability_jump, 3))
                }
                for ee in escalation_events
            ],
            
            # Window analysis
            "num_windows": int(len(windows)),
            "window_probabilities": [float(round(w.ai_probability, 3)) for w in windows],
            
            # Audio quality
            "audio_quality": {
                "snr": float(round(quality_metrics['snr'], 2)),
                "bandwidth": float(round(quality_metrics['bandwidth'], 2)),
                "is_telecom": bool(quality_metrics['is_telecom_quality'])
            },
            
            # Pattern detection
            "patterns": {
                "is_overpolite_ai": bool(polite_metrics['is_overpolite_ai']),
                "energy_variance": float(round(polite_metrics['energy_variance'], 4)),
                "pitch_jitter": float(round(polite_metrics['pitch_jitter'], 2))
            },
            
            # ElevenLabs detection (INTEGRATED)
            "elevenlabs_detection": {
                "enabled": bool(self.use_elevenlabs_detection),
                "probability": float(round(elevenlabs_analysis['elevenlabs_probability'], 3)) if elevenlabs_analysis else None,
                "indicators_count": int(elevenlabs_analysis['tts_indicators_count']) if elevenlabs_analysis else None,
                "is_likely_elevenlabs": bool(elevenlabs_analysis['is_likely_elevenlabs']) if elevenlabs_analysis else None,
                "features": {
                    "prosody_regular": bool(elevenlabs_analysis['features']['prosody_regular']) if elevenlabs_analysis else None,
                    "pauses_regular": bool(elevenlabs_analysis['features']['pauses_regular']) if elevenlabs_analysis else None,
                    "too_clean": bool(elevenlabs_analysis['features']['too_clean']) if elevenlabs_analysis else None,
                    "pitch_smooth": bool(elevenlabs_analysis['features']['pitch_smooth']) if elevenlabs_analysis else None,
                    "no_disfluencies": bool(elevenlabs_analysis['features']['no_disfluencies']) if elevenlabs_analysis else None,
                    "flat_emotion": bool(elevenlabs_analysis['features']['flat_emotion']) if elevenlabs_analysis else None
                } if elevenlabs_analysis else None
            },
            
            # Metadata
            "metadata_suspicious": bool(metadata_suspicious),
            "metadata_note": str(metadata_note) if metadata_note else None,
            
            # Diagnostics
            "audio_duration_seconds": float(round(duration_seconds, 2)),
            "full_audio_duration_seconds": float(round(full_duration_seconds, 2)),
            "inference_latency_ms": float(round(latency_ms, 2)),
            "explanation": str("; ".join(explanation_parts)),
            
            # Detection methods used
            "detection_methods": {
                "embedding_changes": int(embedding_change_count),
                "acoustic_changes": int(acoustic_change_count),
                "method_used": str("acoustic" if acoustic_change_count > embedding_change_count else "embedding")
            },
            
            # Deprecated fields (backward compatibility)
            "detected_language": "N/A",
            "transcription": "Disabled for performance",
            "english_translation": "Disabled",
            "fraud_keywords": [],
            "overall_risk": str(risk_level)
        }
        
        print(f"[TIMING] Total inference: {latency_ms:.2f}ms")
        print(f"[RESULT] {classification} - {risk_level} - AI prob: {final_ai_prob:.2%} - Speaker changes: {speaker_change_count}")
        
        return result


# Global instance
detector_v2 = None

def get_detector_v2(config: Optional[Dict] = None):
    """Get or create the enhanced detector instance"""
    global detector_v2
    if detector_v2 is None:
        detector_v2 = VoiceDetectorV2.get_instance(config)
    return detector_v2