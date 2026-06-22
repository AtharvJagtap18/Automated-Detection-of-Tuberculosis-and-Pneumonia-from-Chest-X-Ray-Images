"""
Flask API for TB Detection
Serves predictions for both HTML frontend and external integrations
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
from pathlib import Path
import os
from werkzeug.utils import secure_filename

# Add backend to path
backend_path = Path(__file__).parent.parent / "BACKEND_FINAL"
sys.path.insert(0, str(backend_path))

from inference import XRayPredictor

# Initialize Flask app
app = Flask(__name__, static_folder='../FRONTEND_FINAL')
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Initialize predictor
print("🔄 Loading TB detection model...")
predictor = XRayPredictor()
print("✅ Model loaded successfully!")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve frontend index page"""
    return send_from_directory('../FRONTEND_FINAL', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('../FRONTEND_FINAL', path)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict disease from uploaded X-ray image
    
    Request:
        - image: X-ray image file
        
    Response:
        {
            "success": true,
            "disease": "Normal|Pneumonia|Tuberculosis",
            "confidence": 0.95,
            "all_probabilities": {
                "Normal": 0.05,
                "Pneumonia": 0.02,
                "Tuberculosis": 0.93
            },
            "tb_stage": "Stage 3" (if TB detected),
            "stage_confidence": 0.87 (if TB detected),
            "inference_time": 2.3,
            "result": "Formatted result string"
        }
    """
    # Check if file is present
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400
    
    file = request.files['image']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Allowed: JPG, PNG, BMP'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        # Get prediction
        result = predictor.predict(str(filepath), return_timing=True)
        
        # Format result string
        disease = result['disease']
        confidence = result['confidence']
        
        result_text = f"🔬 Diagnosis: {disease}\n"
        result_text += f"📊 Confidence: {confidence*100:.1f}%\n\n"
        
        if disease == "Normal":
            result_text += "✅ No significant abnormalities detected.\n"
            result_text += "Recommendation: Routine follow-up."
        elif disease == "Pneumonia":
            result_text += "⚠️ Pneumonia suspected.\n"
            result_text += "Recommendation: Please consult a doctor for clinical evaluation and appropriate treatment."
        else:  # Tuberculosis
            result_text += "🔴 Tuberculosis suspected.\n\n"
            result_text += "Please consult a doctor immediately for:\n"
            result_text += "• Clinical evaluation\n"
            result_text += "• Sputum testing (AFB smear/culture)\n"
            result_text += "• Chest CT scan if needed\n"
            result_text += "• Appropriate TB treatment protocol"
        
        result_text += "\n\n⚠️ This is a decision support tool. Expert medical review required."
        
        # Prepare response
        response = {
            'success': True,
            'disease': disease,
            'confidence': float(confidence),
            'all_probabilities': {k: float(v) for k, v in result.get('all_probabilities', {}).items()},
            'inference_time': result.get('timing', {}).get('total_time', 0),
            'result': result_text
        }
        
        # Clean up uploaded file
        filepath.unlink()
        
        return jsonify(response)
        
    except Exception as e:
        # Clean up on error
        if filepath.exists():
            filepath.unlink()
        
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'ResNet50',
        'version': '1.0'
    })


@app.route('/api/info', methods=['GET'])
def info():
    """Get model information"""
    return jsonify({
        'model': 'ResNet50',
        'task': 'Disease Classification',
        'classes': ['Normal', 'Pneumonia', 'Tuberculosis'],
        'accuracy': '94.64%',
        'auc': '0.9886',
        'inference_time': '2-3 seconds',
        'version': '1.0'
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 TB DETECTION API SERVER")
    print("="*70)
    print("\n📡 Starting Flask server...")
    print("   Frontend: http://localhost:5000")
    print("   API: http://localhost:5000/predict")
    print("   Health: http://localhost:5000/health")
    print("   Info: http://localhost:5000/api/info")
    print("\n" + "="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
