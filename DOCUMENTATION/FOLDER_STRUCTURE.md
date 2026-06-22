# TB Detection System - Organized Folder Structure

## Current Structure (Organized)

```
TB VS NEMO/
│
├── BACKEND/                          # Backend ML Models & API
│   ├── models/
│   │   ├── disease_model.pth        # Trained ResNet50 (90 MB)
│   │   └── stage_model.pth          # TB stage model
│   ├── results/                     # Training results & visualizations
│   │   ├── resnet50_visualizations/ # All graphs & metrics
│   │   └── plots/
│   ├── config.py                    # Configuration
│   ├── preprocess.py                # Image preprocessing
│   ├── models_arch.py               # Model architectures
│   ├── inference.py                 # Prediction engine
│   ├── train_disease.py             # Training script
│   ├── metrics.py                   # Metrics calculation
│   ├── visualizations.py            # Graph generation
│   └── README.md
│
├── FRONTEND/                         # Web Interface
│   ├── index.html                   # Login page
│   ├── main.html                    # Dashboard
│   └── assets/
│       ├── css/
│       │   ├── style.css            # Login styles
│       │   └── main.css             # Dashboard styles
│       └── js/
│           ├── script.js            # Login logic
│           ├── main.js              # Dashboard logic
│           └── theme.js             # Theme switcher
│
├── INTEGRATION/                      # Integration Layer
│   ├── streamlit_app.py             # Streamlit web app
│   ├── flask_api.py                 # Flask REST API
│   ├── requirements_deployment.txt  # Deployment dependencies
│   └── DEPLOYMENT_GUIDE.md          # How to deploy
│
├── DATASET/                          # Dataset & Training Data
│   ├── raw/                         # Downloaded datasets
│   │   ├── tb_dataset/              # TB Chest X-ray Dataset
│   │   └── tbx11k/                  # TBX11K Dataset
│   ├── organized/                   # Organized by class
│   │   ├── Normal/
│   │   ├── Pneumonia/
│   │   └── Tuberculosis/
│   └── splits/                      # Train/val/test splits
│       └── disease_dataset_split.json
│
├── DOCUMENTATION/                    # All Documentation
│   ├── METHODOLOGY.md               # Research methodology
│   ├── ABLATION_STUDY.md            # CNN comparison
│   ├── VISUALIZATIONS_GUIDE.md      # Graph explanations
│   ├── FINAL_MODEL_SUMMARY.md       # Model performance
│   ├── HOW_TO_RUN.md                # Usage instructions
│   └── DEPLOYMENT_GUIDE.md          # Deployment guide
│
└── SCRIPTS/                          # Utility Scripts
    ├── full_pipeline.py             # Complete training pipeline
    ├── ablation_study.py            # Compare CNN architectures
    ├── generate_visualizations.py   # Generate graphs
    └── requirements.txt             # Python dependencies
```

## What Each Folder Contains

### BACKEND/
**Purpose**: Machine learning models and prediction engine

**Key Files**:
- `models/disease_model.pth` - Trained ResNet50 (94.64% accuracy)
- `inference.py` - Main prediction engine
- `config.py` - All configuration settings
- `results/resnet50_visualizations/` - All graphs and metrics

**Use**: Import for predictions
```python
from BACKEND.inference import XRayPredictor
predictor = XRayPredictor()
result = predictor.predict("xray.jpg")
```

### FRONTEND/
**Purpose**: User interface (HTML/CSS/JS)

**Key Files**:
- `index.html` - Login page
- `main.html` - Dashboard with upload
- `assets/js/main.js` - Handles image upload and API calls

**Use**: Serve with Flask or any web server

### INTEGRATION/
**Purpose**: Connect frontend to backend

**Key Files**:
- `streamlit_app.py` - Standalone Streamlit app
- `flask_api.py` - REST API for HTML frontend
- `DEPLOYMENT_GUIDE.md` - How to deploy

**Use**: Run to start web application
```bash
streamlit run INTEGRATION/streamlit_app.py
# OR
python INTEGRATION/flask_api.py
```

### DATASET/
**Purpose**: Training data and splits

**Key Files**:
- `raw/` - Original downloaded datasets (15,906 images)
- `organized/` - Organized by class (32,204 images)
- `splits/` - Train/val/test split information

**Size**: ~5 GB

**Use**: For training and evaluation

---

## How to Use This Structure

### For Development
```bash
# Train model
python SCRIPTS/full_pipeline.py

# Generate visualizations
python SCRIPTS/generate_visualizations.py

# Run Streamlit app
streamlit run INTEGRATION/streamlit_app.py
```

### For Deployment
```bash
# Deploy Streamlit
streamlit run INTEGRATION/streamlit_app.py

# Deploy Flask + HTML
python INTEGRATION/flask_api.py
```

### For Research Paper
```
Use files from:
- BACKEND/results/resnet50_visualizations/ (all graphs)
- DOCUMENTATION/METHODOLOGY.md (methodology section)
- DOCUMENTATION/FINAL_MODEL_SUMMARY.md (results section)
```

---

## Files to Share with Team

### For Frontend Team
```
Share:
- BACKEND/inference.py (prediction API)
- INTEGRATION/flask_api.py (REST API example)
- DOCUMENTATION/DEPLOYMENT_GUIDE.md
```

### For Research/Paper
```
Share:
- BACKEND/results/resnet50_visualizations/ (all graphs)
- DOCUMENTATION/METHODOLOGY.md
- DOCUMENTATION/FINAL_MODEL_SUMMARY.md
- DOCUMENTATION/ABLATION_STUDY.md
```

### For Deployment
```
Share:
- BACKEND/ (entire folder with models)
- FRONTEND/ (entire folder)
- INTEGRATION/streamlit_app.py OR flask_api.py
- INTEGRATION/requirements_deployment.txt
```

---

## Cleanup Old Files

After organizing, you can delete:
- `backend/` (lowercase - now in BACKEND/)
- `tb-detection-frontend/` (now in FRONTEND/)
- Root level scripts (now in SCRIPTS/)
- Root level .md files (now in DOCUMENTATION/)

---

## Current Status

✅ **BACKEND**: Contains trained model (94.64% accuracy)
✅ **FRONTEND**: Your HTML/CSS/JS interface
✅ **INTEGRATION**: Streamlit app running at http://localhost:8501
✅ **DATASET**: 32,204 organized images

**Ready for**: Deployment and team sharing

---

**Last Updated**: February 27, 2026
