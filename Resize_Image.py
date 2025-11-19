import cv2
import numpy as np

# Read image
img = cv2.imread(r"sample_img/9.jpeg")

# 1. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Gaussian blur (optional, improves thresholding)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# 3. Apply threshold
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 4. Resize to MNIST size (28×28)
result = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

# 5. Save output
cv2.imwrite("output.png", result)
