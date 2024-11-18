from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = 'brain_tumor_detector.h5'
model = load_model(MODEL_PATH)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Health check for the API."""
    return jsonify({"message": "Brain Tumor Detection API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    # Validate if an image file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    # Validate file name
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess the image
    try:
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({'error': 'Invalid image content'}), 400

        img = cv2.resize(img, (224, 224))  # Resize to match the model's input size
        img = img / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img)
        result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
        probability = float(prediction[0][0])

        # Return the result
        return jsonify({'prediction': result, 'probability': probability})

    except Exception as e:
        return jsonify({'error': f'Error during processing: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
