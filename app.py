import os
import cv2
import numpy as np
import time
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Global variables for models
baseline_model = None
multiscale_model = None

MODEL_DIR = "models"
IMG_SIZE = (128, 128)
# Alphabetical order as per utils.py logic
CLASSES = ['broken', 'clean', 'dusty', 'rust']

def load_models():
    global baseline_model, multiscale_model
    try:
        baseline_path = os.path.join(MODEL_DIR, "baseline.h5")
        multiscale_path = os.path.join(MODEL_DIR, "multiscale.h5")
        
        if os.path.exists(baseline_path):
            baseline_model = load_model(baseline_path)
            print("Baseline model loaded.")
        else:
            print("Baseline model not found.")
            
        if os.path.exists(multiscale_path):
            multiscale_model = load_model(multiscale_path)
            print("Multi-Scale model loaded.")
        else:
            print("Multi-Scale model not found.")
            
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models on startup
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # RGB
        
        # Preprocess
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3)
        
        results = {}
        
        # Baseline Prediction
        if baseline_model:
            start_time = time.time()
            preds = baseline_model.predict(img)[0]
            class_idx = np.argmax(preds)
            confidence = float(preds[class_idx])
            inference_time = (time.time() - start_time) * 1000 # ms
            
            results['baseline'] = {
                'prediction': CLASSES[class_idx].capitalize(),
                'confidence': confidence,
                'time_ms': round(inference_time, 2)
            }
            
        # Multi-Scale Prediction
        if multiscale_model:
            start_time = time.time()
            preds = multiscale_model.predict(img)[0]
            class_idx = np.argmax(preds)
            confidence = float(preds[class_idx])
            inference_time = (time.time() - start_time) * 1000 # ms
            
            results['multiscale'] = {
                'prediction': CLASSES[class_idx].capitalize(),
                'confidence': confidence,
                'time_ms': round(inference_time, 2)
            }
            
        return jsonify(results)
        
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
