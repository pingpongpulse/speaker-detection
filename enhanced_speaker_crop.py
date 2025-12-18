#!/usr/bin/env python3
# enhanced_speaker_crop.py

import cv2
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
HF_TOKEN_ENV = os.getenv("HF_TOKEN")

from speaker_diarizer import SpeakerDiarizer

import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection

# ------------------------- Face Detection -------------------------
class FaceDetector:
    def __init__(self, min_confidence=0.5):
        self.detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=min_confidence)

    def detect(self, frame) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)
        boxes = []
        if res.detections:
            for d in res.detections:
                bbox = d.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))
                boxes.append((x1, y1, x2, y2))
        return boxes

def find_active_box(boxes: List[Tuple[int, int, int, int]], frame_prev, frame_curr) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    if frame_prev is None:
        return boxes[0]
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
    best_idx = None
    best_score = -1.0
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(gray_curr.shape[1], x2), min(gray_curr.shape[0], y2)
        if x2c <= x1c or y2c <= y1c:
            score = 0.0
        else:
            prev_roi = gray_prev[y1c:y2c, x1c:x2c]
            curr_roi = gray_curr[y1c:y2c, x1c:x2c]
            if prev_roi.shape == curr_roi.shape and prev_roi.size > 0:
                diff = cv2.absdiff(prev_roi, curr_roi)
                score = float(diff.mean()) / 255.0
            else:
                score = 0.0
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score > 0.02:
        return boxes[best_idx]
    return boxes[0] if boxes else None

def timestamp_to_sec(frame_idx: int, fps: float) -> float:
    return frame_idx / fps

# ------------------------- Main Processing -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", help="Input video path")
    parser.add_argument("output_video", help="Output video path")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Sampling fps (frames/sec)")
    parser.add_argument("--token", dest="hf_token", help="HF token (optional for gated models)")
    args = parser.parse_args()

    input_path = Path(args.input_video)
    output_path = Path(args.output_video)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    # Use CLI token if provided, else fallback to .env
    HF_TOKEN = args.hf_token or HF_TOKEN_ENV
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN is not set in CLI or .env file")

    print("[INFO] Loading diarizer...")
    diar = SpeakerDiarizer(hf_token=HF_TOKEN)
    print("[INFO] Running diarization (this extracts audio with ffmpeg)...")
    segments = diar.diarize(str(input_path))
    print(f"[INFO] Diarization returned {len(segments)} segments")

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_fps = args.sample_fps
    sample_interval_frames = max(1, int(round(fps / sample_fps)))

    print(f"[INFO] Video fps: {fps}, sample_fps: {sample_fps}, interval frames: {sample_interval_frames}")

    face_detector = FaceDetector(min_confidence=0.5)

    # ------------------------- First Pass -------------------------
    sampled_active_box_by_frame = {}
    prev_frame = None
    frame_idx = 0
    sampled_count = 0

    print("[INFO] First pass: sampling & detecting faces ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval_frames == 0:
            boxes = face_detector.detect(frame)
            t_sec = timestamp_to_sec(frame_idx, fps)
            active_speaker = None
            for s, e, label in segments:
                if s <= t_sec <= e:
                    active_speaker = label
                    break
            if boxes:
                active_box = find_active_box(boxes, prev_frame, frame)
                if active_speaker is not None or active_box is not None:
                    sampled_active_box_by_frame[frame_idx] = active_box
            prev_frame = frame.copy()
            sampled_count += 1
        frame_idx += 1

    cap.release()
    print(f"[INFO] Found active boxes for {len(sampled_active_box_by_frame)} sampled frames (out of {sampled_count}).")

    # ------------------------- Second Pass -------------------------
    tmp_video = output_path.with_suffix(".noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(tmp_video), fourcc, sample_fps, (width, height))

    cap = cv2.VideoCapture(str(input_path))
    frame_idx = 0
    print("[INFO] Second pass: writing output frames ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        box = sampled_active_box_by_frame.get(frame_idx, None)
        frame_to_write = frame.copy()
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_to_write, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_to_write, f"Frame {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if frame_idx % sample_interval_frames == 0:
            out_writer.write(frame_to_write)
        frame_idx += 1

    cap.release()
    out_writer.release()
    print(f"[INFO] Temporary video written to {tmp_video}")

    # ------------------------- Remux Audio -------------------------
    final_tmp = output_path.with_suffix(".final.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_video),
        "-i", str(input_path),
        "-c", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(final_tmp)
    ]
    print("[INFO] Muxing original audio back into processed video (ffmpeg)...")
    subprocess.run(cmd, check=True)
    final_tmp.replace(output_path)

    try:
        tmp_video.unlink()
    except Exception:
        pass

    print(f"[INFO] Output saved to: {output_path}")
    print("[DONE]")

if __name__ == "__main__":
    main()
