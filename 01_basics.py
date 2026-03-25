"""
01_basics.py — OpenCV Fundamentals
===================================
Topics covered:
  - Reading and displaying an image
  - Understanding BGR color format
  - Accessing pixel values
  - Basic drawing (lines, rectangles, circles, text)
  - Saving an image

Run:
    python 01_basics.py
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# 1. Create a blank canvas (since we have no image file)
# ─────────────────────────────────────────────
# np.zeros → black image  |  shape = (height, width, channels)
canvas = np.zeros((500, 700, 3), dtype=np.uint8)

print("Canvas shape :", canvas.shape)   # (500, 700, 3)
print("Height       :", canvas.shape[0])
print("Width        :", canvas.shape[1])
print("Channels     :", canvas.shape[2])  # 3 = Blue, Green, Red


# ─────────────────────────────────────────────
# 2. Understand BGR (OpenCV uses BGR, NOT RGB)
# ─────────────────────────────────────────────
# Format:  (Blue, Green, Red)  ← opposite of what you expect!
RED    = (0,   0,   255)
GREEN  = (0,   255, 0)
BLUE   = (255, 0,   0)
WHITE  = (255, 255, 255)
YELLOW = (0,   255, 255)
CYAN   = (255, 255, 0)
PURPLE = (255, 0,   255)


# ─────────────────────────────────────────────
# 3. Fill regions with color
# ─────────────────────────────────────────────
# canvas[y1:y2, x1:x2] = color
canvas[0:250, 0:350]   = (30, 30, 60)    # dark blue top-left
canvas[0:250, 350:700] = (30, 60, 30)    # dark green top-right
canvas[250:500, 0:350] = (60, 30, 30)    # dark red bottom-left
canvas[250:500, 350:700] = (40, 40, 40)  # dark gray bottom-right


# ─────────────────────────────────────────────
# 4. Drawing shapes
# ─────────────────────────────────────────────

# Line: cv2.line(image, start_point, end_point, color, thickness)
cv2.line(canvas, (0, 250), (700, 250), WHITE, 2)    # horizontal divider
cv2.line(canvas, (350, 0), (350, 500), WHITE, 2)    # vertical divider

# Rectangle: cv2.rectangle(image, top_left, bottom_right, color, thickness)
# thickness = -1 fills the rectangle
cv2.rectangle(canvas, (50, 50), (150, 150), RED, 3)       # hollow
cv2.rectangle(canvas, (400, 50), (500, 150), YELLOW, -1)  # filled

# Circle: cv2.circle(image, center, radius, color, thickness)
cv2.circle(canvas, (175, 375), 60, CYAN, 3)        # hollow
cv2.circle(canvas, (525, 375), 60, PURPLE, -1)     # filled

# Ellipse: cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thickness)
cv2.ellipse(canvas, (175, 125), (80, 40), 0, 0, 360, GREEN, 2)


# ─────────────────────────────────────────────
# 5. Writing text
# ─────────────────────────────────────────────
# cv2.putText(image, text, origin, font, scale, color, thickness)
# origin = bottom-left corner of text
cv2.putText(canvas, "OpenCV Basics",    (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
cv2.putText(canvas, "Rectangle",        (55,  175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
cv2.putText(canvas, "Filled rect",      (375, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
cv2.putText(canvas, "Hollow circle",   (115, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
cv2.putText(canvas, "Filled circle",   (465, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)


# ─────────────────────────────────────────────
# 6. Accessing and modifying individual pixels
# ─────────────────────────────────────────────
# canvas[y, x] returns [B, G, R]
pixel = canvas[100, 100]
print(f"\nPixel at (100,100): B={pixel[0]}, G={pixel[1]}, R={pixel[2]}")

# Change a 10x10 block of pixels to white
canvas[240:260, 340:360] = (255, 255, 255)


# ─────────────────────────────────────────────
# 7. Save and display
# ─────────────────────────────────────────────
cv2.imwrite("output_basics.png", canvas)
print("\nSaved → output_basics.png")

cv2.imshow("01 - OpenCV Basics (press Q to quit)", canvas)

# waitKey(0) waits forever until a key is pressed
# waitKey(ms) waits for that many milliseconds
key = cv2.waitKey(0)
print(f"Key pressed: {key}")

cv2.destroyAllWindows()
print("Done!")
