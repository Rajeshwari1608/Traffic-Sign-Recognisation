import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model('traffic_sign_model.h5')

# Class names (43 classes in GTSRB)
classes = [f"Class {i}" for i in range(43)]

# Load and preprocess image
img_path = 'test_image.jpg'  # your image path
img = cv2.imread(img_path)
img = cv2.resize(img, (32,32))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
label = np.argmax(pred)
print(f"🚦 Predicted Sign: {classes[label]} (Confidence: {np.max(pred)*100:.2f}%)")
