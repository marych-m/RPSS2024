from roboflow import Roboflow
rf = Roboflow(api_key="lshByBqWKCtNXA5yjxTp")
project = rf.workspace().project("robot-snkuk")
model = project.version(1).model

# infer on a local image
print(model.predict(r'images\test1.jpg', confidence=40, overlap=30).json())

import cv2

# Load the original image
image_path = 'images\\test1.jpg'
original_image = cv2.imread(image_path)

# Get predictions
predictions = model.predict(r'images\test1.jpg', confidence=40, overlap=30).json()['predictions']
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
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Put text with class name and confidence
    text = f'{class_name}: {confidence:.2f}'
    cv2.putText(original_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Count the cubes
    if class_name.lower() == 'orange':
        orange_count += 1
    elif class_name.lower() == 'black':
        black_count += 1

# Display the image with bounding boxes
resized_image = cv2.resize(original_image, (0, 0), fx=0.15, fy=0.15)  # Resize by 15% in both dimensions
cv2.imshow('Object Detection', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the results
print(f'Orange cubes: {orange_count}')
print(f'Black cubes: {black_count}')
print(f'Total cubes: {orange_count + black_count}')