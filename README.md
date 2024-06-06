# Roboterprogramierung: Cube Detection and Pyramid Creation

This project detects orange and black cubes in an image, counts them, and allows the user to create a pyramid with the detected cubes if there are enough cubes of the chosen color.

## Prerequisites

- requisites.txt

## Installation

1. **Clone the repository** (if applicable):
    ```bash
    git clone [https://github.com/your-repository.git](https://github.com/marych-m/RPSS2024)
    cd your-repository
    ```

2. **Install required Python packages**:
    ```bash
    pip install -r /path/to/requirements.txt
    ```

3. **Download your image**:
    - Ensure you have an image with cubes located at `images/test1.jpg`.

## Usage

1. **Run the script**:
    ```bash
    python main.py
    ```

2. **Follow the prompts**:
    - Enter the desired height of the pyramid (1, 2, 3, or 4).
    - Enter the color of the blocks (orange=0 or black=1).

## Code Explanation

### Cube Detection

The script uses a prediction model created through Roboflow to detect cubes in the image and OpenCV to draw bounding boxes around the detected cubes. (More info here: https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/)

1. Initialize the Roboflow model:
    ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="your_api_key")
    project = rf.workspace().project("robot-snkuk")
    model = project.version(1).model
    ```

2. Load the image and get predictions:
    ```python
    image_path = 'images\\test1.jpg'
    original_image = cv2.imread(image_path)
    predictions = model.predict(image_path, confidence=40, overlap=30).json()['predictions']
    ```

3. Count and display the cubes:
    ```python
    orange_count = 0
    black_count = 0
    for prediction in predictions:
        class_name = prediction['class']
        if class_name.lower() == 'orange':
            orange_count += 1
        elif class_name.lower() == 'black':
            black_count += 1

    print(f'Orange cubes: {orange_count}')
    print(f'Black cubes: {black_count}')
    print(f'Total cubes: {orange_count + black_count}')
    ```

### Pyramid Creation

The script allows the user to create a pyramid if there are enough cubes of the specified color.

1. Define the pyramid creation function:
    ```python
    def create_pyramid(hoehe, farbe):
        for i in range(hoehe):
            abstaende = ' ' * (hoehe - i - 1)
            bloecke = farbe * (2 * i + 1)
            print(abstaende + bloecke + abstaende)
    ```

2. Main function to check cube availability and create the pyramid:
    ```python
    def main():
        hoehe = int(input("Geben Sie die Höhe der Pyramide ein (1, 2, 3 oder 4): "))
        farbe = input("Geben Sie die Farbe der Blöcke ein (orange=0 oder schwarz=1): ")

        if farbe == '0':  # Orange
            farbe = 'orange'
            if orange_count < (hoehe * hoehe):
                print("Not enough orange cubes.")
                return
        elif farbe == '1':  # Black
            farbe = 'black'
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
    ```

## Notes

- Make sure to replace `"your_api_key"` with your actual Roboflow API key (I gave mine, so no issue here)
- Adjust the image path if your image is located in a different directory.
- Ensure the class names used in the prediction (e.g., `'orange'`, `'black'`) match the ones defined in your Roboflow project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
