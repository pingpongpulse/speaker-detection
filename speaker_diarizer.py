# speaker_diarizer.py
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is not set in .env file")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except Exception:
    TORCHAUDIO_AVAILABLE = False

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except Exception:
    PYANNOTE_AVAILABLE = False


class SpeakerDiarizer:
    def __init__(self, hf_token: str | None = None):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.pipeline = None

        if not PYANNOTE_AVAILABLE:
            print("[WARNING] pyannote.audio not installed — diarization unavailable.")
            return

        print("[INFO] Attempting to load pyannote pipeline...")
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=self.hf_token
            )
            print("[INFO] pyannote pipeline loaded.")
        except Exception as e:
            print(f"[WARNING] Could not load pyannote pipeline: {e}")
            self.pipeline = None

    def extract_audio(self, video_path: str) -> str:
        """Extract mono 16 kHz WAV using ffmpeg; return wav path."""
        video = Path(video_path)
        wav_path = video.with_suffix(".wav")
        cmd = [
            "ffmpeg", "-y", "-i", str(video),
            "-ar", "16000", "-ac", "1", str(wav_path)
        ]
        print(f"[INFO] Running ffmpeg to extract audio -> {wav_path}")
        subprocess.run(cmd, check=True)
        return str(wav_path)

    def load_audio_as_tensor(self, wav_path: str) -> Dict[str, Any]:
        """Return dict {'waveform': tensor_or_numpy, 'sample_rate': int}"""
        if TORCHAUDIO_AVAILABLE:
            try:
                waveform, sample_rate = torchaudio.load(wav_path)
                return {"waveform": waveform, "sample_rate": int(sample_rate)}
            except Exception as e:
                print(f"[WARNING] torchaudio.load failed: {e}")

        import soundfile as sf
        import numpy as np
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 1:
            data = np.expand_dims(data, 0)
        else:
            data = data.T
        try:
            import torch
            tensor = torch.from_numpy(data)
            return {"waveform": tensor, "sample_rate": int(sr)}
        except Exception:
            return {"waveform": data, "sample_rate": int(sr)}

    def diarize(self, video_path: str) -> List[Tuple[float, float, str]]:
        """Extract audio, run diarization, return list of (start,end,label)."""
        if self.pipeline is None:
            print("[WARNING] No diarization pipeline available — returning empty list.")
            return []

        wav_path = self.extract_audio(video_path)
        audio_dict = self.load_audio_as_tensor(wav_path)

        print("[INFO] Running diarization on preloaded audio...")
        try:
            diarization = self.pipeline(audio_dict)
        except Exception as e:
            print(f"[ERROR] Diarization pipeline call failed: {e}")
            return []

        segments = []
        try:
            # Try pyannote Track API
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append((turn.start, turn.end, speaker))
        except Exception:
            # fallback for dict/list output
            try:
                for record in diarization:
                    start = record.get("start") or record.get("start_time") or record["start"]
                    end = record.get("end") or record.get("end_time") or record["end"]
                    label = record.get("label") or record.get("speaker") or record.get("entity")
                    segments.append((start, end, label))
            except Exception:
                print("[WARNING] Could not parse diarization output format.")

        print(f"[INFO] Diarization complete: {len(segments)} segments.")
        return segments
