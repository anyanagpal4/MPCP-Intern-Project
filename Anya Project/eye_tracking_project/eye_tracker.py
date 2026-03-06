"""
Real-Time Eye Tracking System (Open/Closed classification) using MediaPipe Face Mesh.

Core idea:
- Detect face landmarks
- Extract 6 eye landmarks per eye
- Compute Eye Aspect Ratio (EAR)
- Classify OPEN/CLOSED via threshold
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .ear import calculate_ear


# MediaPipe Face Mesh landmark indices (per project PDF)
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]


Point2D = Tuple[int, int]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def get_eye_landmarks(
    landmarks: Sequence[Any],
    indices: Sequence[int],
    frame_w: int,
    frame_h: int,
) -> list[Point2D]:
    pts: list[Point2D] = []
    for idx in indices:
        lm = landmarks[idx]
        # MediaPipe landmarks are normalized; clamp to avoid rare out-of-range values
        x = int(_clamp01(float(lm.x)) * frame_w)
        y = int(_clamp01(float(lm.y)) * frame_h)
        pts.append((x, y))
    return pts


@dataclass
class EyeState:
    ear: float
    state: str  # "OPEN" | "CLOSED" | "NO FACE"


class EyeTracker:
    def __init__(
        self,
        ear_threshold: float = 0.21,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing: float = 0.0,
        model_path: Optional[str] = None,
    ) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except ModuleNotFoundError as e:
            raise SystemExit(
                "MediaPipe is not installed in this Python environment.\n"
                f"You're running: {sys.executable}\n\n"
                "Fix:\n"
                "  python -m pip install -r eye_tracking_project/requirements.txt\n"
                "Then re-run the program using the SAME python.\n"
            ) from e

        self._mp = mp
        self._mode: str
        self._t0_perf: float = time.perf_counter()

        self.ear_threshold = float(ear_threshold)
        self.smoothing = float(smoothing)
        self._ear_smoothed: Optional[float] = None

        # MediaPipe has two Python APIs in the wild:
        # - Legacy "Solutions" API: mp.solutions.face_mesh.FaceMesh (no external model file needed)
        # - Newer "Tasks" API: FaceLandmarker (requires a .task model file)
        if hasattr(mp, "solutions") and hasattr(getattr(mp, "solutions"), "face_mesh"):
            self._mode = "solutions"
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self._mode = "tasks"
            default_model = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
            if not model_path:
                if os.path.exists(default_model):
                    model_path = default_model
                else:
                    raise SystemExit(
                        "Your installed MediaPipe package does not include the legacy 'mp.solutions.face_mesh' API.\n"
                        f"Detected mediapipe version: {getattr(mp, '__version__', None)}\n\n"
                        "You need a FaceLandmarker model file (.task).\n"
                        "Place it at:\n"
                        f"  {default_model}\n"
                        "or pass:\n"
                        "  --model /path/to/face_landmarker.task\n"
                    )
            model_path = os.path.expanduser(model_path)
            if not os.path.exists(model_path):
                raise SystemExit(f"FaceLandmarker model not found: {model_path}")

            from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (  # type: ignore
                VisionTaskRunningMode,
            )
            from mediapipe.tasks.python.vision.face_landmarker import (  # type: ignore
                FaceLandmarker,
                FaceLandmarkerOptions,
            )

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._mode == "solutions":
            self._face_mesh.close()
        else:
            self._landmarker.close()

    def classify(self, ear: float) -> str:
        return "CLOSED" if ear < self.ear_threshold else "OPEN"

    def _smooth_ear(self, ear: float) -> float:
        if self.smoothing <= 0.0:
            return ear
        if self._ear_smoothed is None:
            self._ear_smoothed = ear
            return ear
        alpha = self.smoothing
        self._ear_smoothed = (1.0 - alpha) * self._ear_smoothed + alpha * ear
        return float(self._ear_smoothed)

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, EyeState]:
        frame_h, frame_w = frame_bgr.shape[:2]

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._mode == "solutions":
            results = self._face_mesh.process(frame_rgb)

            annotated = frame_bgr.copy()
            if not results.multi_face_landmarks:
                return annotated, EyeState(ear=0.0, state="NO FACE")

            face = results.multi_face_landmarks[0]
            landmarks = face.landmark
        else:
            # Tasks API path (FaceLandmarker)
            mp = self._mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms = int((time.perf_counter() - self._t0_perf) * 1000.0)
            result = self._landmarker.detect_for_video(mp_image, ts_ms)

            annotated = frame_bgr.copy()
            if not result.face_landmarks:
                return annotated, EyeState(ear=0.0, state="NO FACE")
            landmarks = result.face_landmarks[0]

        left_eye = get_eye_landmarks(landmarks, LEFT_EYE_IDXS, frame_w, frame_h)
        right_eye = get_eye_landmarks(landmarks, RIGHT_EYE_IDXS, frame_w, frame_h)

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        ear = self._smooth_ear(ear)
        state = self.classify(ear)

        # Draw eye contours + points
        for pts, color in ((left_eye, (255, 255, 0)), (right_eye, (255, 255, 0))):
            cv2.polylines(annotated, [np.array(pts, dtype=np.int32)], isClosed=True, color=color, thickness=1)
            for (x, y) in pts:
                cv2.circle(annotated, (x, y), 2, (0, 255, 255), -1)

        return annotated, EyeState(ear=ear, state=state)


def _draw_hud(
    frame: np.ndarray,
    *,
    frame_idx: int,
    fps: float,
    eye_state: EyeState,
    threshold: float,
) -> np.ndarray:
    out = frame
    h, w = out.shape[:2]

    if eye_state.state == "OPEN":
        color = (0, 200, 0)
    elif eye_state.state == "CLOSED":
        color = (0, 0, 255)
    else:
        color = (0, 165, 255)  # orange for NO FACE

    lines = [
        f"State: {eye_state.state}",
        f"EAR: {eye_state.ear:.3f}",
        f"Threshold: {threshold:.3f}  (+/- to adjust)",
        f"Frame: {frame_idx}",
        f"FPS: {fps:.1f}",
    ]

    x0, y0 = 12, 28
    dy = 24

    # background panel
    panel_h = dy * len(lines) + 12
    panel_w = min(w - 24, 420)
    cv2.rectangle(out, (x0 - 8, y0 - 22), (x0 - 8 + panel_w, y0 - 22 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(out, (x0 - 8, y0 - 22), (x0 - 8 + panel_w, y0 - 22 + panel_h), (60, 60, 60), 1)

    for i, text in enumerate(lines):
        cv2.putText(
            out,
            text,
            (x0, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color if i == 0 else (230, 230, 230),
            2 if i == 0 else 1,
            cv2.LINE_AA,
        )
    return out


def _parse_source(source: str) -> Union[int, str]:
    # If it's an int like "0", treat as webcam index; else treat as path
    try:
        return int(source)
    except ValueError:
        return source


def _open_capture(source: Union[int, str], backend: str) -> cv2.VideoCapture:
    """
    OpenCV backend notes (macOS):
    - Using CAP_AVFOUNDATION is typically the most reliable for webcams.
    """
    if backend == "default":
        return cv2.VideoCapture(source)
    if backend == "avfoundation":
        # Some OpenCV builds on macOS can be flaky opening by index with AVFoundation.
        # Prefer AVFoundation, but gracefully fall back to default if it fails to open.
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
        return cv2.VideoCapture(source)
    if backend == "auto":
        if sys.platform == "darwin" and isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(source)
    raise ValueError(f"Unknown backend: {backend}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time eye tracking (OPEN/CLOSED) using MediaPipe Face Mesh.")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or path to video file.")
    parser.add_argument("--width", type=int, default=640, help="Capture width (webcam only, best-effort).")
    parser.add_argument("--height", type=int, default=480, help="Capture height (webcam only, best-effort).")
    parser.add_argument("--threshold", type=float, default=0.21, help="EAR threshold. EAR < threshold => CLOSED.")
    parser.add_argument("--smoothing", type=float, default=0.0, help="EMA smoothing factor [0..1]. 0 disables.")
    parser.add_argument("--max-faces", type=int, default=1, help="Max faces to detect (we render/process first).")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to FaceLandmarker model (.task). Required when your mediapipe install lacks mp.solutions.face_mesh.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "default", "avfoundation"],
        default="auto",
        help="OpenCV VideoCapture backend. On macOS, 'auto' prefers avfoundation for webcams.",
    )
    parser.add_argument(
        "--probe-cameras",
        action="store_true",
        help="Probe camera indices 0..5 and print which can be opened, then exit.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.smoothing <= 1.0):
        raise SystemExit("--smoothing must be between 0 and 1")

    source = _parse_source(args.source)

    if args.probe_cameras:
        if not isinstance(source, int):
            print("[ERROR] --probe-cameras requires --source to be an integer webcam index (e.g. --source 0).")
            return 2
        print(f"[INFO] Probing camera indices 0..5 (backend={args.backend}) ...")
        any_open = False
        for idx in range(6):
            cap_probe = _open_capture(idx, args.backend)
            ok = cap_probe.isOpened()
            cap_probe.release()
            print(f"  - index {idx}: {'OK' if ok else 'FAIL'}")
            any_open = any_open or ok
        if not any_open and sys.platform == "darwin":
            print("[HINT] If all FAIL on macOS, check System Settings → Privacy & Security → Camera permissions.")
        return 0

    cap = _open_capture(source, args.backend)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
        cap.set(cv2.CAP_PROP_FPS, 30.0)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        if sys.platform == "darwin" and isinstance(source, int):
            print("[HINT] On macOS, ensure camera permissions are granted to your terminal/IDE.")
            print("[HINT] Try: --backend default  (or run: --probe-cameras)")
        return 2

    tracker = EyeTracker(
        ear_threshold=args.threshold,
        max_num_faces=args.max_faces,
        smoothing=args.smoothing,
        model_path=args.model,
    )

    print("[INFO] Starting. Press 'q' to quit. Use +/- to adjust threshold.")

    total_frames = 0
    fps_frames = 0
    t0 = time.perf_counter()
    fps = 0.0
    last_state: Optional[str] = None
    last_print_t = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Frame grab failed / end of video.")
                break

            total_frames += 1
            fps_frames += 1
            annotated, eye_state = tracker.process_frame(frame)

            # FPS estimate (moving average over ~1s)
            now = time.perf_counter()
            dt = now - t0
            if dt >= 1.0:
                fps = fps_frames / dt
                fps_frames = 0
                t0 = now

            annotated = _draw_hud(
                annotated,
                frame_idx=total_frames,
                fps=fps,
                eye_state=eye_state,
                threshold=tracker.ear_threshold,
            )

            cv2.imshow("Real-Time Eye Tracking (MediaPipe + EAR)", annotated)

            # Console status (only when changes, plus periodic heartbeat)
            if eye_state.state != last_state:
                print(f"[STATUS] {eye_state.state} (EAR={eye_state.ear:.3f}, threshold={tracker.ear_threshold:.3f})")
                last_state = eye_state.state
                last_print_t = now
            elif now - last_print_t >= 5.0:
                print(f"[INFO] Running... (EAR={eye_state.ear:.3f}, state={eye_state.state})")
                last_print_t = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("+"), ord("=")):
                tracker.ear_threshold = float(tracker.ear_threshold + 0.005)
            if key in (ord("-"), ord("_")):
                tracker.ear_threshold = float(max(0.0, tracker.ear_threshold - 0.005))

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Clean exit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

