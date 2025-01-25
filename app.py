import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import json
app = Flask(__name__)

# Configure upload folders
app.config['MODEL_UPLOAD_FOLDER'] = 'uploaded_models'
app.config['IMAGE_UPLOAD_FOLDER'] = 'uploaded_images'
os.makedirs(app.config['MODEL_UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_UPLOAD_FOLDER'], exist_ok=True)

# Supported file extensions
ALLOWED_EXTENSIONS_MODEL = {'h5'}
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg', 'tif'}

# In-memory storage for uploaded model
current_model = None
CLASS_NAMES = ['EC', 'SA']  


def allowed_file(filename, allowed_extensions):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess the image."""
    img = Image.open(image_path)#.convert("RGB")  # Convert grayscale to RGB
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Handle model upload."""
    global current_model

    if 'model' not in request.files:
        return jsonify({'error': 'No model file in the request'}), 400

    model_file = request.files['model']
    if model_file.filename == '':
        return jsonify({'error': 'No model selected'}), 400

    if allowed_file(model_file.filename, ALLOWED_EXTENSIONS_MODEL):
        model_path = os.path.join(
            app.config['MODEL_UPLOAD_FOLDER'], secure_filename(model_file.filename)
        )
        model_file.save(model_path)
        current_model = tf.keras.models.load_model(model_path)
        return jsonify({'message': 'Model uploaded and loaded successfully'}), 200

    return jsonify({'error': 'Invalid model file type'}), 400


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make predictions using the uploaded model."""
    global current_model

    if current_model is None:
        return jsonify({'error': 'No model uploaded yet'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if allowed_file(file.filename, ALLOWED_EXTENSIONS_IMAGE):
        filepath = os.path.join(
            app.config['IMAGE_UPLOAD_FOLDER'], secure_filename(file.filename)
        )
        file.save(filepath)

        # Preprocess the image
        img_array = preprocess_image(filepath)

        # Make prediction
        predictions = current_model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        # confidence = round(np.max(predictions) * 100,2)

        confidence = round(np.max(predictions) * 100,2)
        # json_object = json.dumps({'predicted_class': predicted_class, 'confidence': str(confidence)+"%"})

        # return json.dumps({'predicted_class': predicted_class, 'confidence': str(confidence)+"%"}, separators = (',', ': ')),200
        return jsonify({'predicted_class': predicted_class, 'confidence': str(confidence)+"%"}), 200

        # return jsonify({'predicted_class': predicted_class, 'confidence': confidence}), 200

    return jsonify({'error': 'Invalid image file type'}), 400


if __name__ == "__main__":
    app.run(debug=True)
