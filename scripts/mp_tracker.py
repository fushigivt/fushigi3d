#!/usr/bin/env python3
"""MediaPipe Face Landmarker → JSON-over-UDP tracker for Rustuber.

Captures webcam frames, runs MediaPipe Face Landmarker to extract ARKit-compatible
blendshapes and head pose, then sends JSON packets over UDP.

Requirements:
    pip install mediapipe opencv-python

Usage:
    python3 mp_tracker.py --port 12346
"""

import argparse
import json
import math
import socket
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_FILENAME = "face_landmarker.task"


def ensure_model(model_dir: str) -> str:
    """Download the face_landmarker.task model if not already present."""
    path = Path(model_dir) / MODEL_FILENAME
    if path.exists():
        return str(path)
    print(f"Downloading MediaPipe face landmarker model to {path} ...")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, str(path))
    print("Download complete.")
    return str(path)


def rotation_matrix_to_euler(mat: np.ndarray) -> list[float]:
    """Extract pitch/yaw/roll (degrees) from a 4x4 transformation matrix."""
    # mat is row-major 4x4
    r = mat[:3, :3]

    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0

    return [math.degrees(pitch), math.degrees(yaw), math.degrees(roll)]


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaPipe face tracker for Rustuber")
    parser.add_argument("--ip", default="127.0.0.1", help="UDP destination address")
    parser.add_argument("--port", type=int, default=12346, help="UDP destination port")
    parser.add_argument("--capture", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    parser.add_argument("--model-dir", default=".", help="Directory for model file")
    args = parser.parse_args()

    # Import mediapipe after arg parse so --help works without it installed
    import mediapipe as mp

    model_path = ensure_model(args.model_dir)

    # Set up Face Landmarker
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.ip, args.port)

    cap = cv2.VideoCapture(args.capture)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print(f"Error: cannot open camera device {args.capture}", file=sys.stderr)
        sys.exit(1)

    print(f"MediaPipe tracker started: camera={args.capture}, sending to {args.ip}:{args.port}")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Convert BGR→RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.monotonic() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.face_blendshapes:
                packet = {"face_detected": False, "blendshapes": {}, "head_position": [0, 0, 0], "head_rotation": [0, 0, 0]}
            else:
                # Extract blendshapes
                blendshapes = {}
                for bs in result.face_blendshapes[0]:
                    # Skip the neutral category
                    if bs.category_name != "_neutral":
                        blendshapes[bs.category_name] = round(bs.score, 4)

                # Extract head pose from transformation matrix
                head_position = [0.0, 0.0, 0.0]
                head_rotation = [0.0, 0.0, 0.0]
                if result.facial_transformation_matrixes:
                    mat = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
                    head_position = [float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])]
                    head_rotation = rotation_matrix_to_euler(mat)

                packet = {
                    "face_detected": True,
                    "blendshapes": blendshapes,
                    "head_position": [round(v, 4) for v in head_position],
                    "head_rotation": [round(v, 2) for v in head_rotation],
                }

            data = json.dumps(packet).encode("utf-8")
            sock.sendto(data, dest)

    cap.release()
    sock.close()


if __name__ == "__main__":
    main()
