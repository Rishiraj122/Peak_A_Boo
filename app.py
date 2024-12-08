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
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import partial
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# Configure CORS with specific options
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["POST", "GET", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False,
        "max_age": 3600
    }
})

# Configure maximum content length (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

class BackgroundRemover:
    def __init__(self):
        self.selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        os.makedirs('temp', exist_ok=True)
    
    def remove_background(self, image_bytes):
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            # Check image dimensions
            height, width = image.shape[:2]
            if width * height > 4096 * 4096:  # Limit image size
                raise ValueError("Image dimensions too large")
            
            # Process image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.selfie_segmentation.process(image_rgb)
            
            if results.segmentation_mask is None:
                raise ValueError("Failed to segment image")
                
            # Create mask
            mask = np.multiply(results.segmentation_mask > 0.5, 255).astype(np.uint8)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Create RGBA image
            rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = mask
            
            # Clean up
            del image, image_rgb, results, mask
            gc.collect()
            
            return rgba
            
        except Exception as e:
            logger.error(f"Error in remove_background: {str(e)}", exc_info=True)
            raise

def process_image_with_timeout(image_bytes):
    """Process image with a timeout to prevent hanging"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(remover.remove_background, image_bytes)
        try:
            return future.result(timeout=30)  # 30 second timeout
        except TimeoutError:
            logger.error("Image processing timed out")
            raise Exception("Image processing timed out")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

remover = BackgroundRemover()

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Server is running",
        "version": "1.0.0"
    }), 200

@app.route('/segment', methods=['POST', 'OPTIONS'])
def segment_image():
    """Handle image segmentation requests"""
    logger.info(f"Received request: {request.method}")
    logger.info(f"Headers: {dict(request.headers)}")

    if request.method == 'OPTIONS':
        logger.info("Handling OPTIONS request")
        return '', 204
        
    try:
        logger.info("Processing POST request")
        logger.info("Received segmentation request")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        # Validate request
        if 'image' not in request.files:
            logger.error("No image in request")
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        # Check file type
        if not file.content_type.startswith('image/'):
            logger.error(f"Invalid file type: {file.content_type}")
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
            
        logger.info(f"Processing file: {file.filename} ({file.content_type})")
        image_bytes = file.read()
        file_size = len(image_bytes)
        logger.info(f"Image size: {file_size} bytes")
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413
        
        try:
            # Process image with timeout
            result_image = process_image_with_timeout(image_bytes)
            logger.info("Background removal completed")
        except Exception as e:
            logger.error(f"Error in background removal: {str(e)}")
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
        
        try:
            # Convert to PNG and encode
            img = Image.fromarray(result_image)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', optimize=True)
            img_byte_arr = img_byte_arr.getvalue()
            
            encoded = base64.b64encode(img_byte_arr).decode('utf-8')
            logger.info("Image successfully encoded")
            
            # Clean up
            del result_image, img, img_byte_arr
            gc.collect()
            
            return jsonify({'image': encoded})
        except Exception as e:
            logger.error(f"Error in image conversion: {str(e)}")
            return jsonify({'error': f'Conversion error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"General error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

# Add a test endpoint
@app.route('/test', methods=['GET', 'OPTIONS'])
def test():
    """Test endpoint to verify API connectivity"""
    if request.method == 'OPTIONS':
        return '', 204
    logger.info("Test endpoint accessed")
    return jsonify({"status": "ok", "message": "API is accessible"}), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)