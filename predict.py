import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("brain_tumor_model.keras")

# Class labels
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Path to MRI image
image_path = r"C:\Users\balac\OneDrive\Desktop\BrainTumorDetection\Dataset\files\Testing\notumor\Te-no_1.jpg"
# Read image
img = cv2.imread(image_path)

# Resize image
img = cv2.resize(img, (224, 224))

# Normalize
img = img / 255.0

# Convert to batch
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)

predicted_class = classes[np.argmax(prediction)]

print("Prediction:", predicted_class)
