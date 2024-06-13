import cv2
import numpy as np

# Read the image
image = cv2.imread(r'images\test1.jpg')
image = cv2.resize(image, (0, 0), fx=2, fy=2)  # Resize by 15% in both dimensions

# Convert the image to grayscale format
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
ret, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)

# Invert the thresholded image to ensure objects are white on a black background
thresh = cv2.bitwise_not(thresh)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a new image to draw rectangles
rect_image = np.zeros_like(image)

# Draw rectangles around each contour
for contour in contours:
    # Get the minimum area bounding rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # Draw the rectangle on the new image
    cv2.drawContours(rect_image, [box], 0, (255, 255, 255), -1)  # White rectangles

# Resize the image for visualization
resized_image = cv2.resize(rect_image, (0, 0), fx=0.15, fy=0.15)  # Resize by 15% in both dimensions

# Display the result
cv2.imshow('Rectangles Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
