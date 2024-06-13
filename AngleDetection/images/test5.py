import cv2
import numpy as np

def get_rotation_angles(image_path):
    # Read the original image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return None

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
rectangles = get_rotation_angles(image_path)

# Print the final list of detected rectangles data
print(rectangles)
