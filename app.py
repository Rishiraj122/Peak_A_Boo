from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import io
from PIL import Image
import base64
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import partial

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class BackgroundRemover:
    def __init__(self):
        self.selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        os.makedirs('temp', exist_ok=True)
    
    def remove_background(self, image_bytes):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to read image")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.selfie_segmentation.process(image_rgb)
            mask = np.multiply(results.segmentation_mask > 0.5, 255).astype(np.uint8)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = mask
            
            return rgba
            
        except Exception as e:
            print(f"Error in remove_background: {str(e)}")
            raise

def process_image_with_timeout(image_bytes):
    """Process image with a timeout to prevent hanging"""
    with ThreadPoolExecutor() as executor:
        future = executor.submit(remover.remove_background, image_bytes)
        try:
            return future.result(timeout=30)  # 30 second timeout
        except TimeoutError:
            raise Exception("Image processing timed out")

remover = BackgroundRemover()

@app.route('/')
def index():
    # Add a simple health check response
    return jsonify({"status": "healthy", "message": "Server is running"}), 200

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        print("Request received") # Add logging
        
        if 'image' not in request.files:
            print("No image in request")
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
            
        print(f"Processing file: {file.filename}")
        image_bytes = file.read()
        print(f"Image size: {len(image_bytes)} bytes")
        
        try:
            # Use the timeout wrapper here
            result_image = process_image_with_timeout(image_bytes)
            print("Background removal completed")
        except Exception as e:
            print(f"Error in remove_background: {str(e)}")
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
        
        try:
            img = Image.fromarray(result_image)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            encoded = base64.b64encode(img_byte_arr).decode('utf-8')
            print("Image successfully encoded")
            return jsonify({'image': encoded})
        except Exception as e:
            print(f"Error in image conversion: {str(e)}")
            return jsonify({'error': f'Conversion error: {str(e)}'}), 500
            
    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)