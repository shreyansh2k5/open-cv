"""
04_webcam.py — Live Webcam + Frame Tools
==========================================
Topics covered:
  - Webcam capture with proper setup
  - FPS counter
  - Resizing, flipping, mirroring
  - Recording video to file
  - Taking snapshots
  - Multiple camera windows

Run:
    python 04_webcam.py

Controls:
    Q — quit
    S — save snapshot
    R — start/stop recording
    F — toggle flip/mirror
    G — toggle grayscale
    E — toggle edge detection overlay
"""

import cv2
import numpy as np
import time
import os


# ─────────────────────────────────────────────
# FPS Calculator (rolling average — smoother display)
# ─────────────────────────────────────────────
class FPSCounter:
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.times = []

    def update(self):
        self.times.append(time.time())
        if len(self.times) > self.buffer_size:
            self.times.pop(0)

    def get_fps(self):
        if len(self.times) < 2:
            return 0.0
        elapsed = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / elapsed if elapsed > 0 else 0.0


# ─────────────────────────────────────────────
# Video Recorder helper
# ─────────────────────────────────────────────
class VideoRecorder:
    def __init__(self):
        self.writer   = None
        self.is_recording = False
        self.filename = ""

    def start(self, frame_size, fps=20.0):
        self.filename = f"recording_{int(time.time())}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(self.filename, fourcc, fps, frame_size)
        self.is_recording = True
        print(f"Recording started → {self.filename}")

    def write(self, frame):
        if self.is_recording and self.writer:
            self.writer.write(frame)

    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.is_recording = False
        print(f"Recording saved → {self.filename}")


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────
def draw_hud(frame, fps, is_recording, flip, grayscale, edges):
    """Draw a clean heads-up display on the frame."""
    h, w = frame.shape[:2]

    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # FPS
    fps_color = (0, 255, 0) if fps > 25 else (0, 165, 255) if fps > 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

    # Recording indicator (blinking red dot)
    if is_recording:
        if int(time.time() * 2) % 2 == 0:   # blink every 0.5s
            cv2.circle(frame, (w - 25, 25), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 70, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # Active modes
    modes = []
    if flip:      modes.append("FLIP")
    if grayscale: modes.append("GRAY")
    if edges:     modes.append("EDGES")
    if modes:
        mode_text = "  ".join(modes)
        cv2.putText(frame, mode_text, (w // 2 - 60, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 1)

    # Bottom key hint
    hint = "S=snapshot  R=record  F=flip  G=gray  E=edges  Q=quit"
    cv2.putText(frame, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    return frame


def apply_edge_overlay(frame):
    """Detect edges and overlay them in green on the original frame."""
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(blurred, 50, 150)
    # Colour the edges green and blend onto frame
    edge_colored = np.zeros_like(frame)
    edge_colored[:, :, 1] = edges    # only green channel
    return cv2.addWeighted(frame, 0.8, edge_colored, 0.5, 0)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("Webcam Demo")
    print("S=snapshot  R=record  F=flip  G=gray  E=edges  Q=quit")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    # Set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Read actual values (camera may not support requested ones)
    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps:.0f} fps")

    fps_counter  = FPSCounter()
    recorder     = VideoRecorder()
    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    # Toggles
    do_flip      = True
    do_grayscale = False
    do_edges     = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed")
            break

        # ── Transformations (order matters) ────────
        if do_flip:
            frame = cv2.flip(frame, 1)         # 1 = horizontal mirror

        if do_edges:
            frame = apply_edge_overlay(frame)  # must happen before grayscale

        if do_grayscale:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # keep 3 channels

        # ── FPS ────────────────────────────────────
        fps_counter.update()
        fps = fps_counter.get_fps()

        # ── HUD ────────────────────────────────────
        display = draw_hud(frame.copy(), fps,
                           recorder.is_recording,
                           do_flip, do_grayscale, do_edges)

        # ── Record (write clean frame, not HUD) ───
        recorder.write(frame)

        cv2.imshow("04 - Webcam", display)

        # ── Key handling ───────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            filename = os.path.join(snapshot_dir, f"snap_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved → {filename}")

        elif key == ord('r'):
            if recorder.is_recording:
                recorder.stop()
            else:
                recorder.start((actual_w, actual_h))

        elif key == ord('f'):
            do_flip = not do_flip
            print(f"Flip: {'ON' if do_flip else 'OFF'}")

        elif key == ord('g'):
            do_grayscale = not do_grayscale
            print(f"Grayscale: {'ON' if do_grayscale else 'OFF'}")

        elif key == ord('e'):
            do_edges = not do_edges
            print(f"Edge overlay: {'ON' if do_edges else 'OFF'}")

    # ── Cleanup ─────────────────────────────────
    if recorder.is_recording:
        recorder.stop()

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
