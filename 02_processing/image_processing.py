"""
Phase 2 — Image Processing
===========================
Learn how to:
  - Resize and crop
  - Apply blur filters
  - Threshold images
  - Detect edges with Canny
  - Draw shapes and text

Run: python image_processing.py
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# 1. Resize and Crop
# ─────────────────────────────────────────────
def resize_and_crop(img: np.ndarray) -> dict:
    """Resize to fixed size, and crop a region of interest."""
    h, w = img.shape[:2]

    resized_half  = cv2.resize(img, (w // 2, h // 2))
    resized_fixed = cv2.resize(img, (300, 300))

    # Crop: img[y1:y2, x1:x2]
    crop_y1, crop_y2 = h // 4, 3 * h // 4
    crop_x1, crop_x2 = w // 4, 3 * w // 4
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

    print(f"Original:      {img.shape}")
    print(f"Resized half:  {resized_half.shape}")
    print(f"Resized 300px: {resized_fixed.shape}")
    print(f"Cropped:       {cropped.shape}")

    return {
        "Resized half":  resized_half,
        "Resized 300x300": resized_fixed,
        "Cropped":       cropped,
    }


# ─────────────────────────────────────────────
# 2. Blur Filters
# ─────────────────────────────────────────────
def apply_blurs(img: np.ndarray) -> dict:
    """Different blur types — each has different use cases."""
    return {
        # Averages pixels in the kernel (box blur)
        "Blur (average)":    cv2.blur(img, (15, 15)),

        # Weighted average — looks more natural
        "Gaussian blur":     cv2.GaussianBlur(img, (15, 15), 0),

        # Blurs while preserving edges — slower but useful
        "Bilateral filter":  cv2.bilateralFilter(img, 9, 75, 75),

        # Heavy median filter — good for removing salt-and-pepper noise
        "Median blur":       cv2.medianBlur(img, 15),
    }


# ─────────────────────────────────────────────
# 3. Thresholding
# ─────────────────────────────────────────────
def apply_thresholds(img: np.ndarray) -> dict:
    """Convert image to binary (black/white) using different strategies."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Simple: every pixel above 127 → white, else → black
    _, binary       = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Inverted simple threshold
    _, binary_inv   = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Otsu: automatically finds the best threshold value
    _, otsu         = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive: computes threshold per small region (handles uneven lighting)
    adaptive        = cv2.adaptiveThreshold(
                        gray, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2)

    return {
        "Grayscale":    gray,
        "Binary":       binary,
        "Binary inv":   binary_inv,
        "Otsu":         otsu,
        "Adaptive":     adaptive,
    }


# ─────────────────────────────────────────────
# 4. Edge Detection
# ─────────────────────────────────────────────
def detect_edges(img: np.ndarray) -> dict:
    """Find edges using multiple methods."""
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny: two thresholds — low (weak edges), high (strong edges)
    # Start with 50/150 and tune from there
    canny_soft  = cv2.Canny(blurred, 30, 100)
    canny_hard  = cv2.Canny(blurred, 100, 200)

    # Sobel: gradient in X and Y directions
    sobel_x  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_xy = cv2.magnitude(sobel_x, sobel_y)
    sobel_xy = np.uint8(np.clip(sobel_xy, 0, 255))

    return {
        "Canny (soft)": canny_soft,
        "Canny (hard)": canny_hard,
        "Sobel":        sobel_xy,
    }


# ─────────────────────────────────────────────
# 5. Drawing Shapes
# ─────────────────────────────────────────────
def draw_shapes(canvas_size=(500, 700, 3)) -> np.ndarray:
    """Demonstrate all drawing functions on a blank canvas."""
    canvas = np.zeros(canvas_size, dtype=np.uint8)

    # Line: (img, pt1, pt2, color_BGR, thickness)
    cv2.line(canvas, (50, 50), (650, 50), (255, 255, 0), 3)

    # Rectangle: (img, top-left, bottom-right, color, thickness)
    cv2.rectangle(canvas, (50, 80), (250, 200), (0, 255, 0), 2)
    cv2.rectangle(canvas, (280, 80), (480, 200), (0, 0, 255), -1)  # filled

    # Circle: (img, center, radius, color, thickness)
    cv2.circle(canvas, (175, 300), 70, (255, 0, 0), 3)
    cv2.circle(canvas, (400, 300), 70, (0, 200, 200), -1)          # filled

    # Ellipse: (img, center, axes, angle, startAngle, endAngle, color, thickness)
    cv2.ellipse(canvas, (175, 430), (100, 50), 0, 0, 360, (200, 100, 255), 2)

    # Polygon
    pts = np.array([[400, 380], [500, 420], [460, 480], [340, 480], [300, 420]])
    cv2.polylines(canvas, [pts], isClosed=True, color=(100, 255, 100), thickness=2)

    # Text: (img, text, origin, font, scale, color, thickness)
    cv2.putText(canvas, "OpenCV Shapes", (150, 490),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return canvas


# ─────────────────────────────────────────────
# Helper — show multiple windows
# ─────────────────────────────────────────────
def show_all(images: dict) -> None:
    for title, img in images.items():
        cv2.imshow(title, img)
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=== Phase 2: Image Processing ===\n")

    # Build a test image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[50:350, 50:550] = (40, 80, 120)
    cv2.circle(img, (300, 200), 120, (200, 60, 60), -1)
    cv2.rectangle(img, (100, 100), (500, 300), (60, 160, 60), 3)
    cv2.putText(img, "Test Image", (180, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    # --- 1. Resize demos ---
    print("1. Resize & Crop")
    show_all(resize_and_crop(img))

    # --- 2. Blur demos ---
    print("\n2. Blur Filters")
    show_all(apply_blurs(img))

    # --- 3. Threshold demos ---
    print("\n3. Thresholding")
    show_all(apply_thresholds(img))

    # --- 4. Edge detection ---
    print("\n4. Edge Detection")
    show_all(detect_edges(img))

    # --- 5. Shapes ---
    print("\n5. Drawing Shapes")
    shapes_canvas = draw_shapes()
    show_all({"Drawing shapes": shapes_canvas})

    print("\nDone!")


if __name__ == "__main__":
    main()
