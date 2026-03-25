"""
Phase 4 — Live Webcam & Frame Processing
==========================================
Learn how to:
  - Capture live video from webcam
  - Apply real-time effects
  - Show FPS counter
  - Record video to file
  - Handle keyboard shortcuts

Run: python webcam_live.py
"""

import cv2
import numpy as np
import time
from typing import Callable


# ─────────────────────────────────────────────
# Frame effect functions
# Each takes (frame) → modified frame
# ─────────────────────────────────────────────

def effect_none(frame: np.ndarray) -> np.ndarray:
    return frame

def effect_grayscale(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # convert back to 3ch for display

def effect_blur(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(frame, (21, 21), 0)

def effect_edges(frame: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def effect_cartoon(frame: np.ndarray) -> np.ndarray:
    """Cartoon-like effect: strong edges on soft-colored base."""
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.medianBlur(gray, 7)
    edges     = cv2.adaptiveThreshold(blurred, 255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY, 9, 9)
    color     = cv2.bilateralFilter(frame, 9, 250, 250)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_bgr)

def effect_invert(frame: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(frame)

def effect_sepia(frame: np.ndarray) -> np.ndarray:
    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ])
    sepia = cv2.transform(frame, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def effect_pixelate(frame: np.ndarray, block=15) -> np.ndarray:
    h, w  = frame.shape[:2]
    small = cv2.resize(frame, (w // block, h // block),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def effect_emboss(frame: np.ndarray) -> np.ndarray:
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]])
    emboss = cv2.filter2D(gray, -1, kernel) + 128
    return cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)

def effect_sketch(frame: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv     = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch  = cv2.divide(gray, cv2.bitwise_not(blurred), scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────
# FPS counter utility
# ─────────────────────────────────────────────
class FPSCounter:
    def __init__(self, smoothing=10):
        self._times     = []
        self._smoothing = smoothing

    def tick(self) -> float:
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._smoothing:
            self._times.pop(0)

        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ─────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────
def draw_hud(frame: np.ndarray, fps: float,
             effect_name: str, recording: bool) -> np.ndarray:
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # FPS
    color = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Effect name
    cv2.putText(frame, f"Effect: {effect_name}", (130, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Recording indicator
    if recording:
        cv2.circle(frame, (w - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 60, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Controls hint
    hint = "q=quit | e=effect | s=screenshot | r=record | f=flip"
    cv2.putText(frame, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    return frame


# ─────────────────────────────────────────────
# Video recorder
# ─────────────────────────────────────────────
class VideoRecorder:
    def __init__(self, filename: str, fps: float, frame_size: tuple):
        fourcc     = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        print(f"Recording to {filename}")

    def write(self, frame: np.ndarray):
        self._writer.write(frame)

    def release(self):
        self._writer.release()
        print("Recording saved.")


# ─────────────────────────────────────────────
# Main webcam loop
# ─────────────────────────────────────────────
def run_webcam(camera_index: int = 0,
               width: int = 640,
               height: int = 480) -> None:
    """
    Main live webcam loop with effects, FPS counter, screenshot, recording.

    Keyboard shortcuts:
      q         — quit
      e         — cycle through effects
      s         — save screenshot
      r         — toggle video recording
      f         — toggle horizontal flip
      SPACE     — pause/resume
    """
    # ── Effects registry ──────────────────────
    EFFECTS: list[tuple[str, Callable]] = [
        ("Normal",    effect_none),
        ("Grayscale", effect_grayscale),
        ("Blur",      effect_blur),
        ("Edges",     effect_edges),
        ("Cartoon",   effect_cartoon),
        ("Invert",    effect_invert),
        ("Sepia",     effect_sepia),
        ("Pixelate",  effect_pixelate),
        ("Emboss",    effect_emboss),
        ("Sketch",    effect_sketch),
    ]

    # ── Open camera ───────────────────────────
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_w}x{actual_h}")
    print("Controls: q=quit | e=effect | s=screenshot | r=record | f=flip | SPACE=pause")

    # ── State ────────────────────────────────
    fps_counter   = FPSCounter(smoothing=20)
    effect_idx    = 0
    flipped       = False
    paused        = False
    recording     = False
    recorder      = None
    screenshot_n  = 0
    last_frame    = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy() if last_frame is not None else np.zeros(
                (actual_h, actual_w, 3), dtype=np.uint8)

        # Mirror
        if flipped:
            frame = cv2.flip(frame, 1)

        # Apply selected effect
        effect_name, effect_fn = EFFECTS[effect_idx]
        processed = effect_fn(frame.copy())

        # FPS
        fps = fps_counter.tick() if not paused else 0.0

        # Record (before HUD so HUD isn't in the recording)
        if recording and recorder:
            recorder.write(processed)

        # HUD overlay
        display = draw_hud(processed.copy(), fps, effect_name, recording)

        if paused:
            cv2.putText(display, "PAUSED", (actual_w // 2 - 70, actual_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)

        cv2.imshow("OpenCV Webcam", display)

        # ── Key handling ──────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('e'):
            effect_idx = (effect_idx + 1) % len(EFFECTS)
            print(f"Effect: {EFFECTS[effect_idx][0]}")

        elif key == ord('f'):
            flipped = not flipped
            print(f"Flip: {'on' if flipped else 'off'}")

        elif key == ord('s'):
            screenshot_n += 1
            fname = f"screenshot_{screenshot_n:03d}.png"
            cv2.imwrite(fname, processed)
            print(f"Screenshot saved: {fname}")

        elif key == ord('r'):
            if not recording:
                fname    = f"recording_{int(time.time())}.avi"
                recorder = VideoRecorder(fname, 20.0, (actual_w, actual_h))
                recording = True
            else:
                recorder.release()
                recorder  = None
                recording = False

        elif key == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")

    # ── Cleanup ──────────────────────────────
    if recorder:
        recorder.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=== Phase 4: Live Webcam & Frame Processing ===\n")
    run_webcam(camera_index=0, width=640, height=480)


if __name__ == "__main__":
    main()
