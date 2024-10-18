import sys
import numpy as np
import tensorflow as tf
from keras_preprocessing import image
import cv2
from collections import deque

# Force UTF-8 encoding to avoid UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

# Load the pre-trained model
model_path = 'fresh_and_stale_fruit_detector_model.h5'
model = tf.keras.models.load_model(model_path)

# Class labels
siniflar = [
    "fresh_apple",
    "fresh_banana",
    "fresh_bitter_gourd",
    "fresh_capsicum",
    "fresh_orange",
    "fresh_tomato",
    "stale_apple",
    "stale_banana",
    "stale_bitter_gourd",
    "stale_capsicum",
    "stale_orange",
    "stale_tomato"
]

# Freshness index mapping (1-5 for fresh fruits, 0 for stale fruits)
freshness_mapping = {
    "fresh_apple": 5,
    "fresh_banana": 5,
    "fresh_bitter_gourd": 5,
    "fresh_capsicum": 5,
    "fresh_orange": 5,
    "fresh_tomato": 5,
    "stale_apple": 0,
    "stale_banana": 0,
    "stale_bitter_gourd": 0,
    "stale_capsicum": 0,
    "stale_orange": 0,
    "stale_tomato": 0
}

# Initialize a deque to store predictions over multiple frames
prediction_window = deque(maxlen=5)

# Function to preprocess each frame before prediction
def preprocess_image(frame):
    # Resize to match the input shape of the model
    img = cv2.resize(frame, (256, 256))

    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply sharpening filter to enhance edges
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    # Convert BGR to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to an array and normalize pixel values
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    return img_array

# Function to get a prediction with confidence thresholding
def get_prediction(tahmin):
    confidence = np.max(tahmin[0])
    if confidence > 0.7:  # Use a confidence threshold
        return siniflar[np.argmax(tahmin[0])]
    else:
        return "Uncertain"

# Function to get the freshness index for the predicted class
def get_freshness_index(prediction):
    return freshness_mapping.get(prediction, "Unknown")

# Function to stabilize predictions by averaging over multiple frames
def stabilize_prediction():
    if len(prediction_window) == prediction_window.maxlen:
        # Count the occurrences of each prediction in the window
        pred_counts = {pred: prediction_window.count(pred) for pred in set(prediction_window)}
        # Return the prediction with the highest count
        return max(pred_counts, key=pred_counts.get)
    return "Waiting..."

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

# Frame counter to process every nth frame
frame_counter = 0

# Real-time loop for capturing and classifying video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame_counter += 1

    # Process every 5th frame to avoid overload
    if frame_counter % 5 == 0:
        # Preprocess the captured frame
        processed_frame = preprocess_image(frame)

        # Make prediction on the current frame
        tahmin = model.predict(processed_frame)
        prediction = get_prediction(tahmin)

        # Add the prediction to the window for stabilization
        prediction_window.append(prediction)

    # Get the stabilized prediction
    stable_prediction = stabilize_prediction()

    # Get the freshness index for the stabilized prediction
    freshness_index = get_freshness_index(stable_prediction)

    # Display the prediction and freshness index on the frame
    cv2.putText(frame, f"Prediction: {stable_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Freshness Index: {freshness_index}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the video frame with the prediction and freshness index
    cv2.imshow('Live Fruit Detection', frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
