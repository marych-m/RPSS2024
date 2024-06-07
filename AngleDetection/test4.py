import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np


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
    degrees = -int(np.rad2deg(angle)) - 90

    textbox = cv.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    return angle, degrees, label_position


# Read the image
image = cv.imread(r'images\test1.jpg')

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
cv.imwrite("gray.jpg", gray)
# Convert image to binary
_, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imwrite("bw.jpg", bw)
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

for degrees, position in label_positions:
    print("Angle: {} degrees, Position: {}".format(degrees, position))

cv.imshow('Output Image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the output image to the current directory
cv.imwrite("output3.jpg", img)
