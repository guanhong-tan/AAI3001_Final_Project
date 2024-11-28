import os
import sys
import base64
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import tensorflow as tf
import numpy as np
import cv2

# Force UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

UPLOAD_FOLDER = 'C:/Users/bryan/OneDrive/Desktop/Project/AAI3001_Final_Project/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model with error handling
try:
    print("Loading model...")
    model = tf.keras.models.load_model('C:/Users/bryan/OneDrive/Desktop/Project/AAI3001_Final_Project/model/final_model.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

CATEGORIES = ['Glass', 'Cloth', 'Plastic', 'Paper', 'Metal']

BIN_COLORS = {
    'Glass': 'Green',
    'Cloth': 'Green',
    'Plastic': 'Blue',
    'Paper': 'Blue',
    'Metal': 'Dark Blue & White'
}

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
        
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('upload'))
        
        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('upload'))
        
        # Ensure filename is safe
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(file_path)
        session['uploaded_file_path'] = file_path  # Store file path in session
        return redirect(url_for('results'))
    
    return render_template('upload.html')

@app.route('/results', methods=['GET'])
def results():
    file_path = session.get('uploaded_file_path', None)
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('upload'))

    try:
        # Preprocess and predict
        image = preprocess_image(file_path)
        prediction = model.predict(image, verbose=0)
        category = CATEGORIES[np.argmax(prediction)]
        
        # Get the corresponding bin color
        bin_color = BIN_COLORS.get(category, 'Unknown')  # Fallback to 'Unknown' if category is missing

        # Remove the file after processing
        os.remove(file_path)
        
        # Send result to template
        return render_template(
            'results.html', 
            material_type=category, 
            confidence=float(np.max(prediction)),
            is_recyclable=True,
            bin_color=bin_color
        )
    except Exception as e:
        return render_template('results.html', error=str(e))

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    print("Starting Flask application...")
    app.run(debug=True, port=5001, host='127.0.0.1')
