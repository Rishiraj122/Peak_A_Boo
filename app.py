from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import io
from PIL import Image
import base64
import os

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

class BackgroundRemover:
    def __init__(self):
        # Initialize MediaPipe Selfie Segmentation
        self.selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 1 for higher accuracy
        )
        os.makedirs('temp', exist_ok=True)
    
    def remove_background(self, image_bytes):
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to read image")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get segmentation mask
            results = self.selfie_segmentation.process(image_rgb)
            
            # Convert mask to proper format
            mask = np.multiply(results.segmentation_mask > 0.5, 255).astype(np.uint8)
            
            # Optional: Apply slight blur to mask edges for smoother transition
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Create RGBA image
            rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = mask
            
            return rgba
            
        except Exception as e:
            print(f"Error in remove_background: {str(e)}")
            raise

# Initialize model
remover = BackgroundRemover()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Read the image data
        image_bytes = file.read()
        
        # Process image
        try:
            result_image = remover.remove_background(image_bytes)
        except Exception as e:
            print(f"Error in remove_background: {str(e)}")
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
        
        # Convert to PNG and send
        try:
            img = Image.fromarray(result_image)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Convert to base64
            encoded = base64.b64encode(img_byte_arr).decode('utf-8')
            return jsonify({'image': encoded})
        except Exception as e:
            print(f"Error in image conversion: {str(e)}")
            return jsonify({'error': f'Conversion error: {str(e)}'}), 500
            
    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)