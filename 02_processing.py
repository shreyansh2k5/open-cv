"""
02_processing.py — Image Processing Operations
================================================
Topics covered:
  - Grayscale conversion
  - Blurring (Gaussian, Median)
  - Thresholding (Binary, Adaptive, Otsu)
  - Edge detection (Canny)
  - Morphological ops (dilate, erode)
  - Displaying multiple results side by side

Run:
    python 02_processing.py
"""

import cv2
import numpy as np


def show_side_by_side(title, images, labels):
    """Stack multiple images horizontally with labels and display them."""
    # Ensure all images are 3-channel (some ops return grayscale)
    rgb_images = []
    for img in images:
        if len(img.shape) == 2:               # grayscale → convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        rgb_images.append(img)

    # Resize all to same height
    h = 240
    resized = []
    for img in rgb_images:
        ratio = h / img.shape[0]
        w = int(img.shape[1] * ratio)
        resized.append(cv2.resize(img, (w, h)))

    # Add label bar below each image
    labeled = []
    for img, label in zip(resized, labels):
        bar = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
        labeled.append(np.vstack([img, bar]))

    combined = np.hstack(labeled)
    cv2.imshow(title, combined)


# ─────────────────────────────────────────────
# Create a test image (gradient + shapes)
# ─────────────────────────────────────────────
img = np.zeros((400, 600, 3), dtype=np.uint8)

# Gradient background
for i in range(400):
    img[i, :] = [i // 2, 50, 100]

# Shapes to make processing interesting
cv2.rectangle(img, (50, 50),   (200, 200), (0, 200, 50),  -1)
cv2.circle(img,    (350, 150), 80,          (200, 100, 0), -1)
cv2.rectangle(img, (420, 250), (570, 370),  (0, 80, 200),  -1)
cv2.ellipse(img,   (150, 300), (100, 60), 20, 0, 360, (180, 180, 0), -1)

# Add some noise to make blurring visible
noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
img = cv2.add(img, noise)


print("=" * 50)
print("Image Processing Demo")
print("Press any key to move to the next demo")
print("=" * 50)


# ─────────────────────────────────────────────
# 1. COLOR SPACE CONVERSIONS
# ─────────────────────────────────────────────
gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# HSV is used a LOT for color detection (see 03_color_detection.py)
# H = Hue (color 0-179), S = Saturation (0-255), V = Value/Brightness (0-255)

show_side_by_side("1 - Color Spaces",
    [img, gray, hsv],
    ["Original (BGR)", "Grayscale", "HSV"])
print("1. Color spaces — press any key")
cv2.waitKey(0)


# ─────────────────────────────────────────────
# 2. BLURRING
# ─────────────────────────────────────────────
# Gaussian blur — smooths noise, good general-purpose blur
# kernel size must be odd: (3,3), (5,5), (15,15) etc.
blur_small  = cv2.GaussianBlur(img, (5, 5), 0)
blur_medium = cv2.GaussianBlur(img, (15, 15), 0)
blur_large  = cv2.GaussianBlur(img, (31, 31), 0)

# Median blur — great for "salt and pepper" noise, preserves edges better
median = cv2.medianBlur(img, 9)

show_side_by_side("2 - Blurring",
    [img, blur_small, blur_large, median],
    ["Original", "Gaussian 5x5", "Gaussian 31x31", "Median 9x9"])
print("2. Blurring — press any key")
cv2.waitKey(0)


# ─────────────────────────────────────────────
# 3. THRESHOLDING
# ─────────────────────────────────────────────
# Works on grayscale images
# Binary threshold: pixel > thresh → 255, else → 0
_, thresh_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, thresh_inv    = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Otsu's method — automatically finds the best threshold value
_, thresh_otsu   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive threshold — adjusts threshold per local region (good for uneven lighting)
thresh_adaptive  = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,   # neighbourhood size (must be odd)
    C=2             # constant subtracted from mean
)

show_side_by_side("3 - Thresholding",
    [gray, thresh_binary, thresh_otsu, thresh_adaptive],
    ["Grayscale", "Binary (127)", "Otsu auto", "Adaptive"])
print("3. Thresholding — press any key")
cv2.waitKey(0)


# ─────────────────────────────────────────────
# 4. EDGE DETECTION (Canny)
# ─────────────────────────────────────────────
# Always blur first — edges on noisy images are messy
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny(image, lower_threshold, upper_threshold)
# Lower values → more edges detected (more noise too)
# Higher values → only strong edges detected
edges_tight  = cv2.Canny(blurred, 100, 200)   # strict — fewer edges
edges_medium = cv2.Canny(blurred, 50, 150)    # balanced
edges_loose  = cv2.Canny(blurred, 20, 80)     # loose — more edges, more noise

show_side_by_side("4 - Canny Edge Detection",
    [blurred, edges_tight, edges_medium, edges_loose],
    ["Blurred input", "Tight (100-200)", "Medium (50-150)", "Loose (20-80)"])
print("4. Edge detection — press any key")
cv2.waitKey(0)


# ─────────────────────────────────────────────
# 5. MORPHOLOGICAL OPERATIONS
# ─────────────────────────────────────────────
# Work on binary (thresholded) images
# kernel = structuring element (a small matrix of 1s)
kernel = np.ones((5, 5), np.uint8)

# Dilate — expand white regions (fills gaps)
dilated = cv2.dilate(thresh_binary, kernel, iterations=1)

# Erode — shrink white regions (removes small blobs)
eroded = cv2.erode(thresh_binary, kernel, iterations=1)

# Opening = erode then dilate → removes small noise blobs
opened = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, kernel)

# Closing = dilate then erode → fills small holes inside objects
closed = cv2.morphologyEx(thresh_binary, cv2.MORPH_CLOSE, kernel)

show_side_by_side("5 - Morphological Ops",
    [thresh_binary, dilated, eroded, opened],
    ["Binary", "Dilated", "Eroded", "Opened"])
print("5. Morphological ops — press any key")
cv2.waitKey(0)


# ─────────────────────────────────────────────
# 6. PRACTICAL PIPELINE: Noise removal → Edges
# ─────────────────────────────────────────────
step1_gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
step2_blur    = cv2.GaussianBlur(step1_gray, (7, 7), 0)
_, step3_thresh = cv2.threshold(step2_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
step4_edges   = cv2.Canny(step2_blur, 50, 150)
step5_dilate  = cv2.dilate(step4_edges, np.ones((3, 3), np.uint8), iterations=1)

show_side_by_side("6 - Full Pipeline",
    [img, step2_blur, step3_thresh, step5_dilate],
    ["Original", "1. Blur", "2. Threshold", "3. Edges+Dilate"])
print("6. Full pipeline — press any key to exit")
cv2.waitKey(0)


cv2.destroyAllWindows()
print("\nAll processing demos complete!")
print("Next: run 03_color_detection.py")
