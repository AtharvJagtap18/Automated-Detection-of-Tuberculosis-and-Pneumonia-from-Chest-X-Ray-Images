# How to Run TB X-Ray Analysis System

## Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- 8GB RAM minimum
- 5GB free disk space
- Windows/Linux/Mac

---

## Step 1: Install Dependencies

Open terminal/command prompt in the `TB VS NEMO` folder and run:

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (CPU version)
- torchvision
- Pillow
- NumPy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- albumentations

---

## Step 2: Setup Kaggle API (First Time Only)

### 2.1 Get Kaggle API Token
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json` file

### 2.2 Place Kaggle Credentials
**Windows:**
```
C:\Users\YourUsername\.kaggle\kaggle.json
```

**Linux/Mac:**
```
~/.kaggle/kaggle.json
```

---

## Step 3: Train Models (First Time)

### Option A: Full Pipeline (Recommended)
This downloads datasets, organizes data, and trains both models:

```bash
cd "TB VS NEMO"
python full_pipeline.py
```

**Time**: 1-2 hours (10 epochs)  
**Output**: Trained models in `backend/models/`

### Option B: Manual Steps

#### 3.1 Download Datasets
```bash
python -c "import kagglehub; kagglehub.dataset_download('tawsifurrahman/tuberculosis-tb-chest-xray-dataset')"
python -c "import kagglehub; kagglehub.dataset_download('vbookshelf/tbx11k-simplified')"
```

#### 3.2 Organize Data
```bash
python backend/data_preparation.py
```

#### 3.3 Train Disease Model
```bash
python backend/train_disease.py
```

#### 3.4 Train TB Stage Model
```bash
python backend/train_stage.py
```

---

## Step 4: Test Inference

After training completes, test the models:

```bash
python backend/inference.py
```

Or test with your own X-ray image:

```python
from backend.inference import XRayPredictor

predictor = XRayPredictor()
result = predictor.predict("path/to/xray.jpg")

print(f"Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.2%}")

if 'tb_stage' in result:
    print(f"TB Stage: {result['tb_stage']}")
```

---

## Step 5: Use in Your Application

### Python Integration

```python
from backend.inference import XRayPredictor

# Initialize predictor (loads models)
predictor = XRayPredictor()

# Predict on single image
result = predictor.predict("xray_image.jpg")

# Access results
disease = result['disease']  # "Normal", "Pneumonia", or "Tuberculosis"
confidence = result['confidence']  # 0.0 to 1.0

if disease == "Tuberculosis":
    stage = result['tb_stage']  # "Stage 1" to "Stage 5"
    stage_conf = result['stage_confidence']
```

### Batch Processing

```python
import os
from backend.inference import XRayPredictor

predictor = XRayPredictor()

# Process all images in a folder
image_folder = "path/to/xray/images"
results = []

for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        result = predictor.predict(image_path)
        results.append({
            'filename': filename,
            'disease': result['disease'],
            'confidence': result['confidence'],
            'tb_stage': result.get('tb_stage', 'N/A')
        })

# Save results
import json
with open('predictions.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Configuration

Edit `backend/config.py` to customize:

```python
# Model architecture
DISEASE_MODEL_ARCH = "resnet50"  # or "densenet121"
STAGE_MODEL_ARCH = "resnet50"

# Training parameters
NUM_EPOCHS = 10  # Increase for better accuracy
BATCH_SIZE = 16  # Adjust based on RAM
LEARNING_RATE = 0.001

# Inference thresholds
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence
TB_THRESHOLD = 0.6  # TB detection threshold
```

---

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

### Issue: "Kaggle credentials not found"
**Solution**: Setup Kaggle API (see Step 2)

### Issue: "Out of memory"
**Solution**: Reduce batch size in `backend/config.py`
```python
BATCH_SIZE = 8  # or even 4
```

### Issue: "Training too slow"
**Solution**: Reduce epochs or use smaller model
```python
NUM_EPOCHS = 5
DISEASE_MODEL_ARCH = "resnet50"  # Faster than densenet
```

### Issue: "Models not found"
**Solution**: Train models first (Step 3)
```bash
python full_pipeline.py
```

---

## File Structure

```
TB VS NEMO/
├── backend/
│   ├── models/              # Trained models (after training)
│   │   ├── disease_model.pth
│   │   └── stage_model.pth
│   ├── results/             # Training results
│   │   ├── plots/           # Confusion matrices
│   │   └── metrics/         # Performance metrics
│   ├── config.py            # Configuration
│   ├── preprocess.py        # Image preprocessing
│   ├── models_arch.py       # Model architectures
│   ├── inference.py         # Prediction engine
│   ├── train_disease.py     # Disease training
│   ├── train_stage.py       # Stage training
│   ├── evaluate.py          # Evaluation
│   └── metrics.py           # Metrics calculation
├── data/
│   ├── raw/                 # Downloaded datasets
│   ├── organized/           # Organized by class
│   └── splits/              # Train/val/test splits
├── full_pipeline.py         # Complete training pipeline
├── requirements.txt         # Python dependencies
├── METHODOLOGY.md           # Technical methodology
└── HOW_TO_RUN.md           # This file
```

---

## Performance Expectations

### Training (10 epochs)
- **Time**: 1-2 hours on CPU
- **RAM**: 2-4 GB
- **Disk**: 5 GB

### Inference
- **Time**: 2-3 seconds per image
- **RAM**: 1-2 GB
- **Accuracy**: 85-90% (with 10 epochs)

### Production Training (50 epochs)
- **Time**: 4-5 hours on CPU
- **Accuracy**: 90-95%

---

## API Integration Example

### Flask REST API

```python
from flask import Flask, request, jsonify
from backend.inference import XRayPredictor
import os

app = Flask(__name__)
predictor = XRayPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    filepath = f"temp/{file.filename}"
    file.save(filepath)
    
    result = predictor.predict(filepath)
    os.remove(filepath)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI Example

```python
from fastapi import FastAPI, File, UploadFile
from backend.inference import XRayPredictor
import shutil

app = FastAPI()
predictor = XRayPredictor()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filepath = f"temp/{file.filename}"
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = predictor.predict(filepath)
    os.remove(filepath)
    
    return result
```

---

## Next Steps

1. ✅ Train models (Step 3)
2. ✅ Test inference (Step 4)
3. 📦 Package for deployment
4. 🌐 Build frontend/API
5. 🧪 Validate with real data
6. 🚀 Deploy to production

---

## Support

For issues or questions:
1. Check `METHODOLOGY.md` for technical details
2. Review `backend/README.md` for module documentation
3. Check training logs in `backend/results/`

---

**Last Updated**: February 24, 2026  
**Version**: 1.0
