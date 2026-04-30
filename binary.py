import cv2
import numpy as np

# Load image (use one path only)
img = cv2.imread('leaf3.jpg')  

# Check if image loaded
if img is None:
    print("Error: Image not found!")
    exit()
# Show original image
cv2.imshow('Original Image', img)

# -------------------------------
# 1. Convert to Grayscale
# -------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray)

# -------------------------------
# 2. Simple Threshold (Binary)
# -------------------------------
_, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('Simple Threshold', thresh1)

# -------------------------------
# 3. Adaptive Threshold
# -------------------------------
adaptive = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    115, 1
)
cv2.imshow('Adaptive Threshold', adaptive)

# -------------------------------
# 4. Otsu Threshold
# -------------------------------
_, otsu = cv2.threshold(
    gray, 125, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
cv2.imshow('Otsu Threshold', otsu)

# -------------------------------
# Wait & Close
# -------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()