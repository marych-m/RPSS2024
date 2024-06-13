from roboflow import Roboflow
import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np

##Cube Recognition
rf = Roboflow(api_key="lshByBqWKCtNXA5yjxTp")
project = rf.workspace().project("robot-snkuk")
model = project.version(1).model


"""# Initialize the webcam (use 0 for the default camera)
cap = cv.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture a single frame from the webcam
ret, frame = cap.read()

# Release the webcam
cap.release()

# Check if the frame was captured correctly
if not ret:
    print("Error: Could not read frame from webcam.")
    exit()"""

# Define the path to save the captured image
image_path = 'test.jpg'
#image_path = 'CubeRecognition\\images\\test3.jpg'

"""# Save the captured frame to a file
cv.imwrite(image_path, frame)
"""
# Load the saved image
original_image = cv.imread(image_path)
lim1 = 40
lim2 = lim1 + 156
lim3 = 267
lim4 = lim3 + 255
original_image = original_image[lim1:lim2, lim3:lim4]

# infer on a local image
print(model.predict(image_path, confidence=40, overlap=30).json())

# Get predictions
predictions = model.predict(original_image, confidence=40, overlap=30).json()['predictions']
# Initialize an empty list to store the extracted information
prediction_list = []

# Iterate through each prediction
for prediction in predictions:
    # Extract color, x, and y coordinates from the prediction
    color = prediction['class']
    x = prediction['x']
    y = prediction['y']

    # Append color, x, and y coordinates as a tuple to the list
    prediction_list.append((color, x, y))

# Now, `prediction_list` contains color, x, and y coordinates for each prediction
print(prediction_list)

# Initialize counters for orange and black cubes
orange_count = 0
black_count = 0

# Draw bounding boxes on the image
for prediction in predictions:
    x_center, y_center, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
    class_name = prediction['class']
    confidence = prediction['confidence']

    # Calculate top-left corner coordinates
    x = int(x_center - w / 2)
    y = int(y_center - h / 2)
    w=int(w)
    h=int(h)

    # Draw the bounding box
    cv.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Put text with class name and confidence
    text = f'{class_name}: {confidence:.2f}'
    cv.putText(original_image, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Count the cubes
    if class_name.lower() == 'orange':
        orange_count += 1
    elif class_name.lower() == 'black':
        black_count += 1

# Display the image with bounding boxes
resized_image = cv.resize(original_image, (0, 0), fx=0.85, fy=0.85)  # Resize by 15% in both dimensions
cv.imshow('Object Detection', resized_image)
cv.imwrite("sample.jpg", resized_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Print the results
print(f'Orange cubes: {orange_count}')
print(f'Black cubes: {black_count}')
print(f'Total cubes: {orange_count + black_count}')

#############Here begins the new angle code###########################################
# Convert the image to grayscale format
image_gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

# Apply binary thresholding
ret, thresh = cv.threshold(image_gray, 111, 255, cv.THRESH_BINARY)

# Invert the thresholded image to ensure objects are white on a black background
thresh = cv.bitwise_not(thresh)

# Find contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create a new image to draw rectangles
img = np.zeros_like(original_image)

# Draw rectangles around each contour
for contour in contours:
    # Get the minimum area bounding rectangle
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    # Draw the rectangle on the new image
    cv.drawContours(img, [box], 0, (255, 255, 255), -1)  # White rectangles

cv.imwrite('bw.jpg', img)

import cv2
import numpy as np

def get_rotation_angles():
    # Read the original image
    image = img

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store detected rectangles with their center coordinates and angles
    detected_rectangles = []

    for contour in contours:
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # Check if the contour is a quadrilateral (rectangle or square)
        if len(approx) == 4:
            # Get the minimum area bounding rectangle
            rect = cv2.minAreaRect(contour)

            # Get the rotation angle from the rectangle
            angle = rect[2]

            # Correct the angle to be within [0, 180]
            if angle < -45:
                angle = 90 + angle

            # Calculate the center coordinates
            center_x, center_y = int(rect[0][0]), int(rect[0][1])

            # Append rectangle details to the list
            rectangle_data = (angle, center_x, center_y)
            detected_rectangles.append(rectangle_data)

            # Draw the rectangle on the original image
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

            # Draw the center (optional)
            cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

    # Display the image with detected rectangles
    cv2.imshow('Detected Rectangles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected_rectangles

# Example usage
image_path = 'test1 (2).jpg'  # Replace with your image path
rectangles = get_rotation_angles()

# Print the final list of detected rectangles data
print(rectangles)
import math

def merge_lists(prediction_list, rectangles, max_distance=10):
    merged_list = []

    for prediction in prediction_list:
        color, x_pred, y_pred = prediction
        closest_rect = None
        min_distance = float('inf')

        for rect in rectangles:
            angle, x_rect, y_rect = rect
            # Calculate distance between prediction and rectangle
            distance = math.sqrt((x_pred - x_rect) ** 2 + (y_pred - y_rect) ** 2)

            # Check if distance is within the threshold and smaller than the current minimum
            if distance <= max_distance and distance < min_distance:
                closest_rect = rect
                min_distance = distance

        if closest_rect is not None:
            merged_list.append((color, x_pred, y_pred, closest_rect[0]))  # Append angle from closest rectangle
        else:
            merged_list.append((color, x_pred, y_pred, None))  # If no matching rectangle found, append None

    return merged_list

merged_list = merge_lists(prediction_list, rectangles, max_distance=15)  # Adjust max_distance as needed

# Print the merged list
for item in merged_list:
    print(item)

cv.waitKey(0)
cv.destroyAllWindows()

# Save the output image to the current directory
cv.imwrite("output4.jpg", img)

## Pyramid creation functions
def create_pyramid(hoehe, farbe):
    for i in range(hoehe):
        abstaende = ' ' * (hoehe - i - 1)
        bloecke = farbe * (2 * i + 1)
        print(abstaende + bloecke + abstaende)

def main():
    hoehe = int(input("Geben Sie die Höhe der Pyramide ein (1, 2, 3 oder 4): "))
    farbe = input("Geben Sie die Farbe der Blöcke ein (orange=0 oder schwarz=1): ")

    # Check if there are enough cubes
    if farbe == '0':  # Orange
        if orange_count < (hoehe * hoehe):
            print("Not enough orange cubes.")
            return
    elif farbe == '1':  # Black
        if black_count < (hoehe * hoehe):
            print("Not enough black cubes.")
            return
    else:
        print("Invalid color choice.")
        return

    print("Hier ist Ihre Pyramide:")
    create_pyramid(hoehe, farbe)

if __name__ == "__main__":
    main()
