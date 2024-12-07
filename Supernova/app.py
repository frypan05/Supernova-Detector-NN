from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Load the trained model
model = load_model('supernova_detector.h5')

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert('L')  # Convert to grayscale
    img = img.resize((64, 64))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=(0, -1))  # Add batch and channel dimensions
    return img

@app.route('/')
def index():
    return render_template('index.html')  # Render the frontend page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Process the image
        img_data = file.read()
        img = preprocess_image(io.BytesIO(img_data))

        # Make prediction
        prediction = model.predict(img)[0][0]
        threshold = 0.4
        result = "Supernova Detected" if prediction > threshold else "No Supernova Detected"

        return jsonify({'prediction': result, 'confidence': f"{prediction:.4f}"})

if __name__ == '__main__':
    app.run(debug=True)
