# Import necessary libraries
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained CNN model
model = load_model('food_recognition_model.h5')

# Define allowed image file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess uploaded image
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    return img

# Function to classify uploaded image
def classify_image(image):
    prediction = model.predict(image)
    class_id = np.argmax(prediction)
    classes = ['Burger', 'Pizza', 'Sushi', 'Steak', 'Pasta']  # Example classes, modify as needed
    predicted_class = classes[class_id]
    confidence = prediction[0][class_id]
    return predicted_class, confidence

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for image upload and classification
@app.route('/classify', methods=['POST'])
def classify():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', message='No file uploaded')
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return render_template('index.html', message='No file selected')
    
    # Check if file extension is allowed
    if not allowed_file(file.filename):
        return render_template('index.html', message='Invalid file extension')
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)
    
    # Preprocess uploaded image
    image = preprocess_image(file_path)
    
    # Classify image
    predicted_class, confidence = classify_image(image)
    
    # Render template with results
    return render_template('index.html', message='Classification successful', predicted_class=predicted_class, confidence=confidence)

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Run Flask app
    app.run(debug=True)
