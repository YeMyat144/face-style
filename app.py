import io
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
from realtime_main import load_stylization_model, stylize_image_with_model
import os

app = Flask(__name__)
CORS(app)

# Load model once at startup
try:
    model = load_stylization_model('weight.pth')
except FileNotFoundError:
    model = None
    print("weight.pth not found. Skipping model loading.")

# Default route to avoid 404
@app.route('/')
def home():
    return "Anime Face Stylization API is running!"

@app.route('/api/transform', methods=['POST'])
def transform():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    try:
        # Read image file as numpy array
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        pil_image = Image.open(in_memory_file).convert('RGB')
        frame_rgb = np.array(pil_image)

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Stylize
        stylized = stylize_image_with_model(model, frame_rgb)

        # Encode result as JPEG
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(stylized, cv2.COLOR_RGB2BGR))
        return send_file(
            io.BytesIO(img_encoded.tobytes()),
            mimetype='image/jpeg',
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # support Render PORT
    app.run(host='0.0.0.0', port=port, debug=True)
