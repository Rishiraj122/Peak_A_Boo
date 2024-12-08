from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import io
from PIL import Image
import base64
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

class BackgroundRemover:
    def __init__(self):
        # Initialize MediaPipe in a thread-safe way
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
    def remove_background(self, image_bytes):
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to read image")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create a new instance for each request
            with self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1
            ) as selfie_segmentation:
                # Process the image
                results = selfie_segmentation.process(image_rgb)
                
                if results.segmentation_mask is None:
                    raise ValueError("Segmentation failed")
                
                # Convert mask to proper format
                mask = np.multiply(results.segmentation_mask > 0.5, 255).astype(np.uint8)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                
                # Create RGBA image
                rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
                rgba[:, :, 3] = mask
                
                return rgba
            
        except Exception as e:
            logger.error(f"Error in remove_background: {str(e)}")
            raise

# Initialize model
remover = BackgroundRemover()

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        logger.info("Received POST request to /segment")
        logger.info(f"Request headers: {request.headers}")
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Log file info
        logger.info(f"Processing file: {file.filename} ({file.content_type})")
        image_bytes = file.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        # Process image
        try:
            logger.info("Starting background removal")
            result_image = remover.remove_background(image_bytes)
            logger.info("Background removal completed")
        except Exception as e:
            logger.error(f"Error in remove_background: {str(e)}")
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
        
        # Convert to PNG and send
        try:
            img = Image.fromarray(result_image)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Convert to base64
            encoded = base64.b64encode(img_byte_arr).decode('utf-8')
            logger.info("Successfully encoded result")
            return jsonify({'image': encoded})
        except Exception as e:
            logger.error(f"Error in image conversion: {str(e)}")
            return jsonify({'error': f'Conversion error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)