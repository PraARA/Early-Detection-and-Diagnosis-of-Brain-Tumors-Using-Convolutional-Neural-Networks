import tensorflow as tf
import numpy as np
from tensorflow import keras
from pathlib import Path
from PIL import Image

# Load the trained model
model = keras.models.load_model("brain_tumor_model.h5")

# Image size (same as training)
IMG_SIZE = (180, 180)

# Class names (adjust if your dataset labels differ)
class_names = ["No Tumor", "Tumor"]

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Batch of 1
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(f"Prediction: {class_names[np.argmax(score)]}")
    print(f"Confidence: {100 * np.max(score):.2f}%")

# Example usage
if __name__ == "__main__":
    test_image = "datasets/Testing/yes/Y1.jpg"  # change this to your image
    predict_image(test_image)
