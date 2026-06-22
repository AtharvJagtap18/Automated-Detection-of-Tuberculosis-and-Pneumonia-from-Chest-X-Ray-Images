# TB Detection System

AI-powered chest X-ray analysis for tuberculosis screening using ResNet50 deep learning.

## Model Performance

- **Accuracy**: 94.64%
- **AUC**: 0.9886
- **Classes**: Normal, Pneumonia, Tuberculosis
- **Inference Time**: 2-3 seconds per image

## Project Structure

```
TB VS NEMO/
├── BACKEND_FINAL/           # ML models & prediction engine
│   ├── models/
│   │   └── disease_model.pth (90 MB)
│   ├── results/
│   │   └── resnet50_visualizations/  # All graphs & metrics
│   ├── inference.py         # Main prediction API
│   ├── config.py
│   └── ...
│
├── FRONTEND_FINAL/          # Web interface
│   ├── index.html
│   ├── main.html
│   └── assets/
│
├── INTEGRATION_FINAL/       # Deployment apps
│   ├── streamlit_app.py     # Streamlit web app
│   ├── flask_api.py         # Flask REST API
│   └── requirements_deployment.txt
│
├── DATASET_FINAL/           # Training data (32,204 images)
│   ├── organized/
│   └── splits/
│
├── SCRIPTS/                 # Training scripts
│   └── full_pipeline.py
│
└── DOCUMENTATION/           # Research docs
```

## Quick Start

### Run Streamlit App (Recommended)
```bash
streamlit run INTEGRATION_FINAL/streamlit_app.py
```
Opens at: http://localhost:8501

### Run Flask API + HTML Frontend
```bash
python INTEGRATION_FINAL/flask_api.py
```
Opens at: http://localhost:5000

## Installation

```bash
pip install -r INTEGRATION_FINAL/requirements_deployment.txt
```

## How It Works

1. Upload chest X-ray image
2. ResNet50 analyzes the image
3. Get prediction: Normal, Pneumonia, or Tuberculosis
4. View confidence scores
5. Follow clinical recommendations

## Visualizations

All graphs and metrics available in:
`BACKEND_FINAL/results/resnet50_visualizations/`

- Training curves (Loss & Accuracy vs Epoch)
- Confusion matrix
- ROC curves with AUC scores
- Classification report

## Deployment

### Streamlit Cloud (Free)
1. Push to GitHub
2. Deploy at https://share.streamlit.io
3. Select `INTEGRATION_FINAL/streamlit_app.py`

### Railway.app (Production)
1. Push to GitHub
2. Deploy at https://railway.app
3. Runs Flask API with HTML frontend

## Medical Disclaimer

This is a **decision support tool** for screening purposes only. All predictions must be reviewed by qualified medical professionals. Not approved for clinical diagnosis without expert validation.

## Documentation

See `DOCUMENTATION/` folder for detailed methodology and research documentation.

## Capstone Project

TB Detection System using Deep Learning
ResNet50 Architecture | 94.64% Accuracy | 0.9886 AUC

---

**Last Updated**: February 27, 2026
