import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load test data
with open('dataset/test.pickle', 'rb') as f:
    test_data = pickle.load(f)

X_test, y_test = test_data['features'], test_data['labels']

X_test = X_test / 255.0
num_classes = len(np.unique(y_test))
y_test = to_categorical(y_test, num_classes)

# Load model
model = load_model('traffic_sign_model.h5')

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")
