import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing import image
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.engine.sequential import Sequential
import cv2

# Load the model
model_path = 'fresh_and_stale_fruit_detector_model.h5'
model = tf.keras.models.load_model(model_path)

# Load and preprocess the image
image_path = "stale_apple.jpg"
img = image.load_img(image_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Get the prediction
prediction = model.predict(img_array)

# Extract the single prediction value
prediction_value = prediction[0][0]

# Map the prediction to a scale from 1 to 5
# Assuming the model output is a probability of being fresh
# You can adjust the thresholds based on your model's performance
if prediction_value < 0.2:
    freshness_scale = 1  # Very stale
elif prediction_value < 0.4:
    freshness_scale = 2  # Stale
elif prediction_value < 0.6:
    freshness_scale = 3  # Neutral
elif prediction_value < 0.8:
    freshness_scale = 4  # Fresh
else:
    freshness_scale = 5  # Very fresh

print(f"The fruit is rated {freshness_scale} on a scale from 1 (very stale) to 5 (very fresh).")