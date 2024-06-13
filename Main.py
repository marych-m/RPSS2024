from roboflow import Roboflow
import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np

##Cube Recognition
rf = Roboflow(api_key="lshByBqWKCtNXA5yjxTp")
project = rf.workspace().project("robot-snkuk")
model = project.version(1).model


# Initialize the webcam (use 0 for the default camera)
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
    exit()

# Define the path to save the captured image
image_path = 'captured_image.jpg'
#image_path = 'CubeRecognition\\images\\test3.jpg'

# Save the captured frame to a file
cv.imwrite(image_path, frame)

# Load the saved image
original_image = cv.imread(image_path)

# infer on a local image
print(model.predict(image_path, confidence=40, overlap=30).json())

# Load the original image
original_image = cv.imread(image_path)

# Get predictions
predictions = model.predict(r'captured_image.jpg', confidence=40, overlap=30).json()['predictions']
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
resized_image = cv.resize(original_image, (0, 0), fx=0.15, fy=0.15)  # Resize by 15% in both dimensions
cv.imshow('Object Detection', resized_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Print the results
print(f'Orange cubes: {orange_count}')
print(f'Black cubes: {black_count}')
print(f'Total cubes: {orange_count + black_count}')

##Angles around Z-axis
def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    # Calculate the angle between the two points
    angle = atan2(p[1] - q[1], p[0] - q[0])

    # Ensure the x-axis is the most horizontal option and goes to the right
    if abs(angle) > pi / 2:
        if angle > 0:
            angle -= pi
        else:
            angle += pi

    # Calculate the hypotenuse
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)

    # Draw the line
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    # Create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 2)

    # Determine which eigenvector represents the most vertical line
    if abs(eigenvectors[0, 1]) > abs(eigenvectors[1, 1]):
        # Use eigenvector 0 as p1
        p1 = (
            cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
            cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0]
        )
        p2 = (
            cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
            cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0]
        )
    else:
        # Use eigenvector 1 as p1
        p1 = (
            cntr[0] + 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
            cntr[1] + 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0]
        )
        p2 = (
            cntr[0] - 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
            cntr[1] - 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0]
        )

    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    ## [visualization]

    # Label with the rotation angle
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    label_position = (cntr[0], cntr[1])  # Position of the label
    degrees = abs(-int(np.rad2deg(angle)) - 90)

    textbox = cv.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    return angle, degrees, label_position

# Read the image
image = cv.imread(image_path)

# Convert the image to grayscale format
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply binary thresholding
ret, thresh = cv.threshold(image_gray, 120, 255, cv.THRESH_BINARY)

# Invert the thresholded image to ensure objects are white on a black background
thresh = cv.bitwise_not(thresh)

# Find contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create a new image to draw rectangles
img = np.zeros_like(image)

# Draw rectangles around each contour
for contour in contours:
    # Get the minimum area bounding rectangle
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    # Draw the rectangle on the new image
    cv.drawContours(img, [box], 0, (255, 255, 255), -1)  # White rectangles

cv.imshow('Rectangles Image', img)

# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite("gray4.jpg", gray)
# Convert image to binary
_, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imwrite("bw4.jpg", bw)
# Find all the contours in the thresholded image
contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# Store labels and positions
label_positions = []

for i, c in enumerate(contours):

    # Calculate the area of each contour
    area = cv.contourArea(c)

    # Ignore contours that are too small or too large
    if area < 90000 or 10000000 < area:
        continue

    # Draw each contour only for visualisation purposes
    cv.drawContours(img, contours, i, (0, 0, 255), 2)

    # Find the orientation of each shape
    angle, degrees, label_position = getOrientation(c, img)
    label_positions.append((degrees, label_position))  # Store angle and label position
print(label_positions)
for degrees, position in label_positions:
    print("Angle: {} degrees, Position: {}".format(degrees, position))

# Define the maximum search radius
max_radius = 3000  # Adjust as needed

# Initialize an empty list to store the combined results
combined_list = []

# Iterate through each element in prediction_list
for prediction in prediction_list:
    # Extract the coordinates of the prediction
    x_pred, y_pred = prediction[1], prediction[2]

    # Iterate through each element in label_positions
    for label_position in label_positions:
        # Extract the coordinates of the label_position
        x_label, y_label = label_position[1]

        # Initialize the search radius
        radius = 1

        # Search in a gradually increasing radius until a match is found or max_radius is reached
        while radius <= max_radius:
            # Check if the coordinates are sufficiently close
            if abs(x_pred - x_label) < radius and abs(y_pred - y_label) < radius:
                # If the coordinates are close, combine the elements
                combined_list.append(prediction + (label_position[0],))
                break  # Exit the inner loop after finding a match

            # Increment the search radius
            radius += 1

        # If a match is found, exit the outer loop
        if len(combined_list) > len(prediction_list):
            break

    # If a match is found, exit the outer loop
    if len(combined_list) > len(prediction_list):
        break

# Now, `combined_list` contains the combined elements from prediction_list and label_positions
print(combined_list)

cv.imshow('Output Image', img)
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
