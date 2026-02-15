"""
AI Voice Detector Application

A multilingual fraud detection system with:
- V1: Original AI/Human classifier
- V2: Enhanced detector with multi-speaker analysis, escalation detection, and multilingual support

Version: 2.0.0
"""

__version__ = '2.0.0'
__author__ = 'Your Team'
__description__ = 'Multilingual AI Voice Detection API with enhanced fraud detection'

# Package metadata
APP_NAME = "AI Voice Detector"
API_VERSION = "2.0.0"

# Import main components for easy access
from .models import VoiceDetector, VoiceDetectorV2

__all__ = [
    'VoiceDetector',
    'VoiceDetectorV2',
]

# Package configuration
class Config:
    """Application configuration"""
    
    # API Settings
    API_TITLE = "AI Voice Detector API"
    API_DESCRIPTION = "Multilingual fraud detection API with V1 (original) and V2 (enhanced) detectors"
    API_VERSION = "2.0.0"
    
    # Model Settings
    DEFAULT_DETECTOR = "v2"  # 'v1' or 'v2'
    V2_CHECKPOINT_PATH = "checkpoints/best.pt"
    
    # Training Settings
    TRAINING_DATA_DIR = "dataset/training_data"
    CHECKPOINT_DIR = "checkpoints"
    
    # Performance Settings
    MAX_AUDIO_DURATION = 30.0  # seconds
    TARGET_LATENCY_MS = 500
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Version info function
def get_version_info():
    """Get detailed version information"""
    return {
        "app_version": __version__,
        "api_version": API_VERSION,
        "detectors": {
            "v1": {
                "model": "facebook/wav2vec2-base",
                "features": ["AI/Human classification", "Basic fraud detection"]
            },
            "v2": {
                "model": "facebook/wav2vec2-xls-r-300m",
                "features": [
                    "Multilingual support (Hindi/Tamil/Telugu/Malayalam)",
                    "Speaker change detection",
                    "Escalation detection",
                    "Multi-speaker analysis",
                    "Enhanced risk scoring",
                    "Telecom audio handling",
                    "Short clip stabilization"
                ]
            }
        }
    }

# Convenience initialization function
def init_detectors(load_v1=True, load_v2=True, v2_checkpoint=None):
    """
    Initialize detectors on application startup.
    
    Args:
        load_v1: Whether to load V1 detector
        load_v2: Whether to load V2 detector
        v2_checkpoint: Path to V2 checkpoint (optional)
    
    Returns:
        dict with 'v1' and/or 'v2' detector instances
    
    Example:
        >>> from app import init_detectors
        >>> detectors = init_detectors(load_v2=True, v2_checkpoint='checkpoints/best.pt')
        >>> result = detectors['v2'].detect_fraud('audio.wav')
    """
    import logging
    logger = logging.getLogger(__name__)
    
    detectors = {}
    
    if load_v1:
        try:
            logger.info("Loading V1 detector...")
            detectors['v1'] = VoiceDetector.get_instance()
            logger.info("✓ V1 detector loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load V1 detector: {e}")
    
    if load_v2:
        try:
            logger.info("Loading V2 detector...")
            
            if v2_checkpoint:
                # Try to load with trained checkpoint
                from .training.utils import load_trained_classifier
                detectors['v2'] = load_trained_classifier(v2_checkpoint)
                logger.info(f"✓ V2 detector loaded with trained checkpoint: {v2_checkpoint}")
            else:
                # Load without checkpoint (untrained)
                detectors['v2'] = VoiceDetectorV2.get_instance()
                logger.warning("⚠ V2 detector loaded WITHOUT trained checkpoint (predictions will be random)")
                logger.warning("  → Train with: python app/training/train_classifier.py --data_dir dataset/training_data")
                
        except Exception as e:
            logger.error(f"✗ Failed to load V2 detector: {e}")
    
    return detectors

# Banner for CLI/debugging
def print_banner():
    """Print application banner"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          AI VOICE DETECTOR - Version {__version__}                  ║
║                                                              ║
║  Multilingual fraud detection with enhanced features        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Features:
  ✓ Multilingual support (Hindi/Tamil/Telugu/Malayalam)
  ✓ Speaker change detection
  ✓ Escalation pattern detection
  ✓ Multi-speaker analysis
  ✓ Sub-500ms latency

Detectors:
  • V1: Original detector (wav2vec2-base)
  • V2: Enhanced detector (XLS-R multilingual)

Documentation: /docs/
"""
    print(banner)