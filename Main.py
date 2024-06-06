from roboflow import Roboflow
import cv2

# Initialize the Roboflow model
rf = Roboflow(api_key="lshByBqWKCtNXA5yjxTp")
project = rf.workspace().project("robot-snkuk")
model = project.version(1).model

# Path to the image
image_path = 'CubeRecognition\\images\\test1.jpg'

# Get predictions
predictions = model.predict(image_path, confidence=40, overlap=30).json()['predictions']

# Initialize counters for orange and black cubes
orange_count = 0
black_count = 0

# Load the original image
original_image = cv2.imread(image_path)

# Draw bounding boxes on the image and count the cubes
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
    cv2.putText(original_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8)

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

# Pyramid creation functions
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
