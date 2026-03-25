"""
Phase 1 — Image Basics
======================
Learn how to:
  - Read an image from disk
  - Display it in a window
  - Convert between color spaces
  - Save a modified image

Run: python image_basics.py
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# 1. Read an image
# ─────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    """Load image from disk. OpenCV reads in BGR (not RGB)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    print(f"Loaded image — shape: {img.shape}, dtype: {img.dtype}")
    # shape = (height, width, channels)
    return img


# ─────────────────────────────────────────────
# 2. Create a sample image (when no file available)
# ─────────────────────────────────────────────
def create_sample_image() -> np.ndarray:
    """Create a 400x600 BGR image with colored rectangles."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Draw colored blocks (BGR format!)
    img[0:200, 0:200]   = (255, 0, 0)    # Blue block
    img[0:200, 200:400] = (0, 255, 0)    # Green block
    img[0:200, 400:600] = (0, 0, 255)    # Red block
    img[200:400, 0:300] = (0, 255, 255)  # Yellow block
    img[200:400, 300:]  = (255, 0, 255)  # Magenta block

    # Write text on the image
    cv2.putText(img, "OpenCV Basics", (150, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return img


# ─────────────────────────────────────────────
# 3. Color space conversions
# ─────────────────────────────────────────────
def explore_color_spaces(img: np.ndarray) -> dict:
    """Convert image to various color spaces."""
    conversions = {
        "grayscale": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "rgb":       cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        "hsv":       cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
    }
    print("Color space shapes:")
    for name, converted in conversions.items():
        print(f"  {name}: {converted.shape}")
    return conversions


# ─────────────────────────────────────────────
# 4. Display images
# ─────────────────────────────────────────────
def display_images(images: dict) -> None:
    """Show multiple images in separate windows."""
    for title, img in images.items():
        cv2.imshow(title, img)

    print("\nPress any key to close all windows...")
    cv2.waitKey(0)             # 0 = wait forever until key press
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 5. Save image
# ─────────────────────────────────────────────
def save_image(img: np.ndarray, output_path: str) -> None:
    """Save image to disk."""
    success = cv2.imwrite(output_path, img)
    if success:
        print(f"Saved image to: {output_path}")
    else:
        print(f"Failed to save image to: {output_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=== Phase 1: Image Basics ===\n")

    # Create a sample image (no file needed)
    img = create_sample_image()
    print(f"Image shape: {img.shape}")   # (height, width, channels)
    print(f"Image dtype: {img.dtype}")   # uint8 — values 0–255

    # Explore channels
    b, g, r = cv2.split(img)
    print(f"\nBlue channel  min/max: {b.min()} / {b.max()}")
    print(f"Green channel min/max: {g.min()} / {g.max()}")
    print(f"Red channel   min/max: {r.min()} / {r.max()}")

    # Convert color spaces
    conversions = explore_color_spaces(img)

    # Display original + grayscale
    display_images({
        "Original (BGR)": img,
        "Grayscale":      conversions["grayscale"],
        "HSV":            conversions["hsv"],
    })

    # Save output
    save_image(img, "output_sample.png")


if __name__ == "__main__":
    main()
