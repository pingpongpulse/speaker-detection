import soundfile as sf
import numpy as np

from speaker_diarizer import SpeakerDiarizer

# load audio
audio, sr = sf.read("audio.wav")   # sr must be 16000 ideally

# init diarizer
diarizer = SpeakerDiarizer(hf_token="hf_xxx_your_token_here")

# run
segments = diarizer(audio, sr)

for s in segments:
    print(s)
