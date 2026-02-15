"""
Models package for AI Voice Detection.

This package contains:
- VoiceDetector (V1): Original detector with basic AI/Human classification
- VoiceDetectorV2 (V2): Enhanced detector with multi-speaker, escalation detection, and multilingual support
"""

from .detector import VoiceDetector
from .detector_v2 import VoiceDetectorV2

__all__ = [
    'VoiceDetector',      # V1: Original detector
    'VoiceDetectorV2',    # V2: Enhanced detector
]

__version__ = '2.0.0'

# Convenience functions
def get_detector_v1():
    """Get V1 detector instance (original)"""
    return VoiceDetector.get_instance()

def get_detector_v2(config=None):
    """Get V2 detector instance (enhanced)"""
    return VoiceDetectorV2.get_instance(config=config)