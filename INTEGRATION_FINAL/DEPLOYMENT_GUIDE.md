# TB Detection System - Deployment Guide

## Overview

You have two deployment options:
1. **Streamlit** - Quick, Python-based web app (recommended for demos)
2. **Flask + HTML Frontend** - Production-ready with your custom frontend

Both use the same ResNet50 backend model (94.64% accuracy).

---

## Option 1: Streamlit Deployment (Easiest)

### Local Deployment

1. **Install dependencies**:
```bash
cd "TB VS NEMO"
pip install -r requirements_deployment.txt
```

2. **Run Streamlit app**:
```bash
streamlit run streamlit_app.py
```

3. **Access app**:
- Open browser: http://localhost:8501
- Upload X-ray image
- Get instant predictions

### Streamlit Cloud Deployment (Free)

1. **Push to GitHub**:
```bash
git init
git add .
git commit -m "TB detection system"
git push origin main
```

2. **Deploy on Streamlit Cloud**:
- Go to https://streamlit.io/cloud
- Sign in with GitHub
- Click "New app"
- Select your repository
- Set main file: `streamlit_app.py`
- Click "Deploy"

3. **Configuration**:
- Streamlit will automatically install dependencies from `requirements_deployment.txt`
- Model files must be included in repository (90 MB)
- Free tier: Unlimited public apps

**Limitations**:
- ⚠️ Model file (90 MB) must be in GitHub (max 100 MB per file)
- ⚠️ Free tier has resource limits
- ⚠️ May be slow on free tier

---

## Option 2: Flask + HTML Frontend (Production)

### Local Deployment

1. **Install dependencies**:
```bash
cd "TB VS NEMO"
pip install -r requirements_deployment.txt
```

2. **Run Flask server**:
```bash
python flask_api.py
```

3. **Access app**:
- Open browser: http://localhost:5000
- Your custom HTML frontend will load
- Upload X-ray and get predictions

### Vercel Deployment (Recommended for Production)

Vercel doesn't support PyTorch models directly. Use these alternatives:

#### Alternative A: Vercel Frontend + Separate Backend

**Frontend on Vercel:**
1. Deploy HTML frontend to Vercel
2. Point API calls to separate backend server

**Backend on:**
- Railway.app (supports Python/PyTorch)
- Render.com (free tier available)
- Heroku (paid)
- Your own server

#### Alternative B: Railway.app (Full Stack)

Railway supports Python and large files:

1. **Create `railway.json`**:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python flask_api.py",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

2. **Create `Procfile`**:
```
web: python flask_api.py
```

3. **Deploy**:
- Go to https://railway.app
- Sign in with GitHub
- Click "New Project" → "Deploy from GitHub"
- Select your repository
- Railway will auto-detect Python and deploy

**Pricing**: $5/month for hobby plan

---

## Option 3: Docker Deployment (Most Flexible)

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements_deployment.txt .
RUN pip install --no-cache-dir -r requirements_deployment.txt

# Copy application
COPY backend/ backend/
COPY tb-detection-frontend/ tb-detection-frontend/
COPY flask_api.py .
COPY streamlit_app.py .

# Expose port
EXPOSE 5000

# Run Flask API
CMD ["python", "flask_api.py"]
```

### Build and Run

```bash
# Build image
docker build -t tb-detection .

# Run container
docker run -p 5000:5000 tb-detection
```

### Deploy to Cloud

**Docker-compatible platforms:**
- Google Cloud Run
- AWS ECS/Fargate
- Azure Container Instances
- DigitalOcean App Platform
- Fly.io

---

## Recommended Deployment Strategy

### For Demo/Presentation
**Use: Streamlit Local**
```bash
streamlit run streamlit_app.py
```
- ✅ Fastest setup (1 command)
- ✅ Beautiful UI out of the box
- ✅ Perfect for demos

### For Production/Capstone
**Use: Railway.app (Flask + HTML)**
```bash
# Push to GitHub
git push origin main

# Deploy on Railway
# Connect GitHub repo
# Railway auto-deploys
```
- ✅ Your custom frontend
- ✅ Professional appearance
- ✅ Scalable
- ✅ $5/month

### For Enterprise
**Use: Docker + Cloud Run**
- ✅ Fully containerized
- ✅ Auto-scaling
- ✅ Production-grade
- ✅ Pay-per-use pricing

---

## File Structure for Deployment

```
TB VS NEMO/
├── backend/
│   ├── models/
│   │   └── disease_model.pth (90 MB) ⚠️ Required
│   ├── config.py
│   ├── preprocess.py
│   ├── models_arch.py
│   └── inference.py
├── tb-detection-frontend/
│   ├── index.html
│   ├── main.html
│   └── assets/
│       ├── css/
│       └── js/
├── streamlit_app.py
├── flask_api.py
├── requirements_deployment.txt
└── README.md
```

**Critical**: The `disease_model.pth` file (90 MB) must be included!

---

## Testing Deployment

### Test Streamlit
```bash
streamlit run streamlit_app.py
```
- Upload test X-ray
- Verify prediction appears
- Check confidence scores

### Test Flask API
```bash
python flask_api.py
```

**Test with curl:**
```bash
curl -X POST -F "image=@test_xray.jpg" http://localhost:5000/predict
```

**Test in browser:**
- Go to http://localhost:5000
- Login (any username/password)
- Upload X-ray
- Verify results

---

## Environment Variables

Create `.env` file for production:

```env
FLASK_ENV=production
MODEL_PATH=backend/models/disease_model.pth
UPLOAD_FOLDER=uploads
MAX_FILE_SIZE=16777216
PORT=5000
```

---

## Performance Optimization

### For Faster Inference

1. **Use GPU** (if available):
```python
# In backend/config.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

2. **Batch Processing**:
```python
# Process multiple images at once
results = predictor.predict_batch(image_paths)
```

3. **Model Quantization**:
```python
# Reduce model size and speed up inference
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## Security Considerations

### For Production Deployment

1. **Authentication**:
```python
# Add proper authentication (not just frontend)
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()
```

2. **Rate Limiting**:
```python
from flask_limiter import Limiter
limiter = Limiter(app, default_limits=["100 per hour"])
```

3. **Input Validation**:
```python
# Validate image format and size
# Check for malicious files
# Sanitize filenames
```

4. **HTTPS**:
- Use SSL certificates
- Enforce HTTPS in production

---

## Monitoring

### Add Logging

```python
import logging

logging.basicConfig(
    filename='tb_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.route('/predict', methods=['POST'])
def predict():
    logging.info(f"Prediction request from {request.remote_addr}")
    # ... prediction code
    logging.info(f"Result: {disease}, Confidence: {confidence}")
```

### Track Metrics

```python
# Track prediction statistics
predictions_count = 0
tb_detected_count = 0
average_confidence = []
```

---

## Troubleshooting

### Issue: Model not found
**Solution**: Ensure `backend/models/disease_model.pth` exists
```bash
python full_pipeline.py  # Train model if missing
```

### Issue: Out of memory
**Solution**: Reduce batch size or use smaller model
```python
# In backend/config.py
BATCH_SIZE = 8
```

### Issue: Slow inference
**Solution**: Use GPU or optimize model
```python
DEVICE = "cuda"  # If GPU available
```

### Issue: CORS errors
**Solution**: Already handled with flask-cors
```python
CORS(app)  # Allows frontend to call API
```

---

## Quick Start Commands

### Streamlit (Demo)
```bash
pip install -r requirements_deployment.txt
streamlit run streamlit_app.py
```

### Flask (Production)
```bash
pip install -r requirements_deployment.txt
python flask_api.py
```

### Docker
```bash
docker build -t tb-detection .
docker run -p 5000:5000 tb-detection
```

---

## Next Steps

1. ✅ Test locally (Streamlit or Flask)
2. ✅ Verify predictions are correct
3. ✅ Choose deployment platform
4. ✅ Deploy to cloud
5. ✅ Share URL with team

---

**Recommended**: Start with Streamlit for quick demo, then move to Flask + Railway for production.

**Last Updated**: February 27, 2026
