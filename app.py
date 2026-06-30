from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load trained model
model = load_model("brain_tumor_model.keras")

# Class labels
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']

    if file.filename == "":
        return "No image selected"

    # Save uploaded image
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Read image
    img = cv2.imread(filepath)

    if img is None:
        return "Unable to read image"

    # Resize image
    img = cv2.resize(img, (224, 224))

    # Normalize image
    img = img / 255.0

    # Convert to batch
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]

    # Confidence
    confidence = float(np.max(prediction) * 100)

    # Tumor status
    if predicted_class == "notumor":
        message = "✅ No Tumor Detected"
    else:
        message = "⚠️ Tumor Detected"

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence, 2),
        message=message,
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)
