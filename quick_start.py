#!/usr/bin/env python3
"""
Quick start guide for enhanced speaker detection with frame sampling and speaker focus
"""

import sys
from pathlib import Path

def print_usage():
    print("""
🎯 Enhanced Speaker Detection - Quick Start Guide

=== FRAME SAMPLING & SPEAKER FOCUS FEATURES ===

1. DEMO VERSION (No pyannote.audio required)
   Shows frame sampling and speaker focus in action:
   
   python demo_enhanced_crop.py input_video.mp4 output.mp4
   
   Options:
   --sample-fps 2.0    Frame sampling rate (default: 2.0 fps)

2. FULL VERSION (Requires pyannote.audio)
   Complete pipeline with diarization:
   
   python enhanced_speaker_crop.py input_video.mp4 output.mp4 --token YOUR_HF_TOKEN

3. ENHANCED ORIGINAL (Backward compatible)
   Enhanced version of original crop_to_speaker.py:
   
   python crop_to_speaker_enhanced.py input_video.mp4 output.mp4 --token YOUR_HF_TOKEN

=== KEY FEATURES ===

✅ Frame Sampling: Analyze 2 fps instead of every frame (15x faster!)
✅ Speaker Focus: Movement + lighting analysis to identify active speaker
✅ Smooth Interpolation: Seamless cropping between sampled frames
✅ 16:9 Aspect Ratio: Professional output format
✅ Backward Compatibility: Original behavior available with --disable-enhancements

=== PERFORMANCE METRICS ===

From test with parrot1.mp4 (22.8s, 684 frames):
- Total frames: 684
- Sampled frames: 49 (7.2% of total)
- Processing efficiency: 92.8% reduction in computation
- Output: Smooth, speaker-focused video

=== USAGE EXAMPLES ===

# Basic usage with frame sampling
python demo_enhanced_crop.py interview.mp4 focused.mp4

# Higher precision (more sampled frames)
python demo_enhanced_crop.py meeting.mp4 focused.mp4 --sample-fps 5.0

# Full pipeline with diarization (requires HF token)
python enhanced_speaker_crop.py presentation.mp4 focused.mp4 --token hf_your_token

# Backward compatibility mode
python crop_to_speaker_enhanced.py video.mp4 output.mp4 --token hf_token --disable-enhancements

=== REQUIREMENTS ===

Install dependencies:
pip install -r requirements.txt

Key packages:
- opencv-python (face detection)
- mediapipe (face tracking)
- tqdm (progress bars)
- pyannote.audio (speaker diarization, full version only)

=== TROUBLESHOOTING ===

❌ torchaudio errors: Use demo version or install compatible torchaudio
❌ Memory issues: Reduce --sample-fps value
❌ No faces detected: Ensure good lighting in video
❌ Choppy output: Increase --sample-fps or check face detection confidence

=== FILES CREATED ===

- demo_enhanced_crop.py: Demo with frame sampling (no diarization)
- enhanced_speaker_crop.py: Full version with diarization
- crop_to_speaker_enhanced.py: Enhanced original script
- README_ENHANCED.md: Detailed documentation

Happy speaker detection! 🎬
""")

if __name__ == "__main__":
    print_usage()