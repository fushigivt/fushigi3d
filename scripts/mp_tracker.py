#!/usr/bin/env python3
"""MediaPipe Face & Pose Landmarker → JSON-over-UDP tracker for Rustuber.

Captures webcam frames, runs MediaPipe Face Landmarker to extract ARKit-compatible
blendshapes and head pose, and optionally runs Pose Landmarker for body tracking.
Sends combined JSON packets over UDP.

Requirements:
    pip install mediapipe opencv-python

Usage:
    python3 mp_tracker.py --port 12346
    python3 mp_tracker.py --port 12346 --no-pose
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

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
POSE_MODEL_FILENAME = "pose_landmarker_lite.task"

# MediaPipe Pose world landmark indices we care about:
# 11=left_shoulder, 12=right_shoulder, 13=left_elbow, 14=right_elbow,
# 15=left_wrist, 16=right_wrist, 23=left_hip, 24=right_hip
POSE_LANDMARK_MAP = {
    11: "leftShoulder",
    12: "rightShoulder",
    13: "leftElbow",
    14: "rightElbow",
    15: "leftWrist",
    16: "rightWrist",
    23: "leftHip",
    24: "rightHip",
}


def ensure_model(model_dir: str, url: str, filename: str) -> str:
    """Download a model file if not already present."""
    path = Path(model_dir) / filename
    if path.exists():
        return str(path)
    print(f"Downloading MediaPipe model to {path} ...")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(path))
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
    parser.add_argument("--no-pose", action="store_true", help="Disable body pose tracking")
    args = parser.parse_args()

    # Import mediapipe after arg parse so --help works without it installed
    import mediapipe as mp

    face_model_path = ensure_model(args.model_dir, MODEL_URL, MODEL_FILENAME)

    # Set up Face Landmarker
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    # Optionally set up Pose Landmarker
    pose_landmarker = None
    use_pose = not args.no_pose
    if use_pose:
        pose_model_path = ensure_model(args.model_dir, POSE_MODEL_URL, POSE_MODEL_FILENAME)
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=VisionRunningMode.VIDEO,
            output_segmentation_masks=False,
            num_poses=1,
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

    pose_status = "enabled" if use_pose else "disabled"
    print(f"MediaPipe tracker started: camera={args.capture}, sending to {args.ip}:{args.port}, pose={pose_status}")

    # Create landmarkers using context managers
    face_ctx = FaceLandmarker.create_from_options(face_options)
    face_lm = face_ctx.__enter__()

    pose_ctx = None
    pose_lm = None
    if use_pose:
        pose_ctx = PoseLandmarker.create_from_options(pose_options)
        pose_lm = pose_ctx.__enter__()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Convert BGR→RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.monotonic() * 1000)

            # Face detection
            face_result = face_lm.detect_for_video(mp_image, timestamp_ms)

            if not face_result.face_blendshapes:
                packet = {
                    "face_detected": False,
                    "blendshapes": {},
                    "head_position": [0, 0, 0],
                    "head_rotation": [0, 0, 0],
                }
            else:
                # Extract blendshapes
                blendshapes = {}
                for bs in face_result.face_blendshapes[0]:
                    # Skip the neutral category
                    if bs.category_name != "_neutral":
                        blendshapes[bs.category_name] = round(bs.score, 4)

                # Extract head pose from transformation matrix
                head_position = [0.0, 0.0, 0.0]
                head_rotation = [0.0, 0.0, 0.0]
                if face_result.facial_transformation_matrixes:
                    mat = np.array(face_result.facial_transformation_matrixes[0]).reshape(4, 4)
                    head_position = [float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])]
                    head_rotation = rotation_matrix_to_euler(mat)

                packet = {
                    "face_detected": True,
                    "blendshapes": blendshapes,
                    "head_position": [round(v, 4) for v in head_position],
                    "head_rotation": [round(v, 2) for v in head_rotation],
                }

            # Pose detection
            if pose_lm is not None:
                pose_result = pose_lm.detect_for_video(mp_image, timestamp_ms)

                if pose_result.pose_world_landmarks and len(pose_result.pose_world_landmarks) > 0:
                    world_lms = pose_result.pose_world_landmarks[0]
                    body_landmarks = {}
                    for idx, name in POSE_LANDMARK_MAP.items():
                        if idx < len(world_lms):
                            lm = world_lms[idx]
                            # MediaPipe Pose world: X=right, Y=up, Z=toward camera
                            # glTF: X=right, Y=up, Z=toward viewer
                            # Negate Z to convert to glTF coordinates
                            body_landmarks[name] = [
                                round(float(lm.x), 4),
                                round(float(lm.y), 4),
                                round(float(-lm.z), 4),
                            ]
                    packet["body_detected"] = True
                    packet["body_landmarks"] = body_landmarks
                else:
                    packet["body_detected"] = False
                    packet["body_landmarks"] = {}
            else:
                packet["body_detected"] = False
                packet["body_landmarks"] = {}

            data = json.dumps(packet).encode("utf-8")
            sock.sendto(data, dest)
    finally:
        face_ctx.__exit__(None, None, None)
        if pose_ctx is not None:
            pose_ctx.__exit__(None, None, None)
        cap.release()
        sock.close()


if __name__ == "__main__":
    main()
