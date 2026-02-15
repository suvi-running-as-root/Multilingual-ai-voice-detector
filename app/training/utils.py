"""
Utilities for loading trained checkpoints and testing
"""

import torch
# Update this import to use relative import
from ..models.detector_v2 import VoiceDetectorV2


def load_trained_classifier(checkpoint_path: str, device: str = 'cpu'):
    """
    Load a trained classifier checkpoint into VoiceDetectorV2
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., 'checkpoints/best.pt')
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        detector: VoiceDetectorV2 instance with trained classifier
    """
    # Initialize detector
    detector = VoiceDetectorV2.get_instance()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load classifier weights
    detector.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    # Set to eval mode
    detector.classifier.eval()
    
    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"  - Val AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
    
    return detector




def test_on_audio_file(detector, audio_path: str):
    """
    Test the detector on a single audio file
    
    Args:
        detector: VoiceDetectorV2 instance
        audio_path: Path to audio file
    
    Returns:
        result: Detection result dictionary
    """
    result = detector.detect_fraud(audio_path)
    
    print("\n" + "="*60)
    print(f"Testing: {audio_path}")
    print("="*60)
    print(f"Classification: {result['classification']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"AI Probability: {result['ai_probability']:.2%}")
    print(f"Confidence: {result['confidence_score']:.2%}")
    print(f"\nSpeaker Changes: {result['speaker_changes']}")
    print(f"Has Escalation: {result['has_escalation']}")
    print(f"Audio Duration: {result['audio_duration_seconds']}s")
    print(f"Inference Latency: {result['inference_latency_ms']:.2f}ms")
    print(f"\nExplanation: {result['explanation']}")
    print("="*60)
    
    return result


def batch_test(detector, audio_files: list):
    """
    Test the detector on multiple audio files
    
    Args:
        detector: VoiceDetectorV2 instance
        audio_files: List of audio file paths
    
    Returns:
        results: List of detection results
    """
    from tqdm import tqdm
    
    results = []
    correct = 0
    total = 0
    
    print(f"\nTesting on {len(audio_files)} files...")
    
    for audio_file in tqdm(audio_files):
        result = detector.detect_fraud(audio_file)
        results.append({
            'file': audio_file,
            'result': result
        })
    
    # Print summary
    print("\n" + "="*60)
    print("Batch Test Summary")
    print("="*60)
    
    ai_count = sum(1 for r in results if r['result']['classification'] == 'AI')
    human_count = len(results) - ai_count
    
    print(f"Total files: {len(results)}")
    print(f"  - Classified as AI: {ai_count}")
    print(f"  - Classified as Human: {human_count}")
    print(f"  - Average latency: {sum(r['result']['inference_latency_ms'] for r in results) / len(results):.2f}ms")
    
    return results


# Example usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python utils.py <checkpoint_path> <audio_file>")
        print("Example: python utils.py checkpoints/best.pt test_audio.wav")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    audio_path = sys.argv[2]
    
    # Load detector with trained classifier
    detector = load_trained_classifier(checkpoint_path)
    
    # Test on audio file
    result = test_on_audio_file(detector, audio_path)
