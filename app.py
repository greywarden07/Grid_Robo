from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from collections import deque

app = Flask(__name__)

# Load the model
model_path = 'fresh_and_stale_fruit_detector_model.h5'
model = tf.keras.models.load_model(model_path)

# Define the classes
siniflar = ['Fresh', 'Stale']

# Initialize a deque to store predictions for stabilization
prediction_window = deque(maxlen=10)

def preprocess_image(img):
    img = cv2.resize(img, (256, 256))
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

# Function to map prediction to a freshness scale
def map_to_freshness_scale(tahmin):
    prediction_value = tahmin[0][0]
    if prediction_value < 0.2:
        return 1  # Very stale
    elif prediction_value < 0.4:
        return 2  # Stale
    elif prediction_value < 0.6:
        return 3  # Neutral
    elif prediction_value < 0.8:
        return 4  # Fresh
    else:
        return 5  # Very fresh

# Function to stabilize predictions by averaging over multiple frames
def stabilize_prediction():
    if len(prediction_window) == prediction_window.maxlen:
        # Count the occurrences of each prediction in the window
        pred_counts = {pred: prediction_window.count(pred) for pred in set(prediction_window)}
        # Return the prediction with the highest count
        return max(pred_counts, key=pred_counts.get)
    return "Waiting..."

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_array = preprocess_image(frame)
        tahmin = model.predict(img_array)
        prediction = get_prediction(tahmin)
        freshness_scale = map_to_freshness_scale(tahmin)

        # Add the prediction to the window for stabilization
        prediction_window.append(prediction)

        # Get the stabilized prediction
        stabilized_prediction = stabilize_prediction()

        # Display the result
        cv2.putText(frame, f"Prediction: {stabilized_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Freshness Scale: {freshness_scale}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)