# TB X-Ray Analysis System - Methodology

## 1. Problem Statement

Develop an offline, CPU-optimized deep learning system for automated tuberculosis (TB) detection and staging from chest X-ray images. The system must:
- Classify chest X-rays into Normal, Pneumonia, or Tuberculosis
- For TB cases, determine disease stage (Stage 1-5)
- Run entirely offline on standard laptops without GPU
- Provide predictions in under 3 seconds per image

## 2. Dataset

### 2.1 Data Sources
Two publicly available Kaggle datasets were used:

1. **TB Chest X-ray Dataset** (tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
   - 4,203 chest X-ray images
   - Contains TB and Normal cases

2. **TBX11K Simplified** (vbookshelf/tbx11k-simplified)
   - 11,703 chest X-ray images
   - Large-scale TB screening dataset

**Total Dataset**: 15,906 chest X-ray images

### 2.2 Data Organization
Images were organized into three disease classes:
- **Normal**: Healthy chest X-rays
- **Pneumonia**: Pneumonia cases (augmented from available data)
- **Tuberculosis**: TB-positive cases

### 2.3 Data Split
- **Training Set**: 70% (11,134 images)
- **Validation Set**: 15% (2,386 images)
- **Test Set**: 15% (2,386 images)

Stratified splitting was used to maintain class distribution across all sets.

## 3. System Architecture

### 3.1 Two-Stage Prediction Pipeline

The system uses a cascaded two-model approach:

```
Input X-Ray Image
       ↓
[Stage 1: Disease Classifier]
       ↓
Normal / Pneumonia / TB
       ↓
   If TB detected
       ↓
[Stage 2: TB Stage Classifier]
       ↓
Stage 1 / 2 / 3 / 4 / 5
```

### 3.2 Model Architecture

Both models use **ResNet50** architecture with transfer learning:

**Disease Classifier**:
- Base: ResNet50 pretrained on ImageNet
- Input: 224×224×3 RGB images
- Output: 3 classes (Normal, Pneumonia, TB)
- Final layer: Fully connected (2048 → 3)

**TB Stage Classifier**:
- Base: ResNet50 pretrained on ImageNet
- Input: 224×224×3 RGB images
- Output: 5 classes (Stage 1-5)
- Final layer: Fully connected (2048 → 5)

**Why ResNet50?**
- Proven performance on medical imaging
- Efficient on CPU (23M parameters)
- Good balance between accuracy and speed
- Strong transfer learning capabilities

## 4. Preprocessing Pipeline

### 4.1 Image Preprocessing
1. **Resize**: 224×224 pixels (ResNet50 input size)
2. **Normalization**: ImageNet mean and std
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
3. **Grayscale to RGB**: Convert single-channel X-rays to 3-channel

### 4.2 Data Augmentation (Training Only)
- **Horizontal Flip**: 50% probability
- **Rotation**: ±10 degrees
- **Brightness**: 0.9-1.1 range
- **Contrast**: 0.9-1.1 range
- **Gaussian Noise**: Variance 10-50
- **Zoom**: ±10%

Augmentation improves model generalization and reduces overfitting.

## 5. Training Strategy

### 5.1 Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 16 (CPU-optimized)
- **Epochs**: 10 (configurable)
- **Loss Function**: Cross-Entropy with class weights
- **Device**: CPU (for offline deployment)

### 5.2 Class Imbalance Handling
Class weights calculated using inverse frequency:
```
weight_i = 1 / (count_i + ε)
normalized_weight_i = weight_i / Σ(weights) × num_classes
```

This ensures minority classes (e.g., Pneumonia) receive higher importance during training.

### 5.3 Learning Rate Scheduling
- **Scheduler**: ReduceLROnPlateau
- **Mode**: Minimize validation loss
- **Factor**: 0.5 (halve learning rate)
- **Patience**: 5 epochs
- **Purpose**: Adaptive learning rate for better convergence

### 5.4 Early Stopping
- **Patience**: 10 epochs
- **Metric**: Validation loss
- **Purpose**: Prevent overfitting, save training time

### 5.5 Model Checkpointing
- Save best model based on validation accuracy
- Checkpoint includes:
  - Model weights
  - Optimizer state
  - Training history
  - Configuration metadata

## 6. Evaluation Metrics

### 6.1 Primary Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### 6.2 Visualization
- **Confusion Matrix**: Class-wise prediction analysis
- **Training Curves**: Loss and accuracy over epochs
- **Per-Class Metrics**: Precision, recall, F1 for each class

## 7. Inference Pipeline

### 7.1 Prediction Flow
```python
1. Load and preprocess X-ray image
2. Run Disease Classifier
3. Get disease prediction and confidence
4. If TB detected with confidence > 60%:
   - Run TB Stage Classifier
   - Get stage prediction and confidence
5. Return results with timing information
```

### 7.2 Confidence Thresholds
- **Disease Classification**: 50% minimum confidence
- **TB Stage Classification**: 60% minimum confidence
- Low confidence predictions flagged for manual review

### 7.3 Performance Targets
- **Inference Time**: < 3 seconds per image on CPU
- **Memory Usage**: < 2GB RAM
- **Model Size**: < 100MB per model

## 8. Technology Stack

### 8.1 Core Libraries
- **PyTorch**: Deep learning framework
- **torchvision**: Pretrained models and transforms
- **Pillow**: Image loading and processing
- **NumPy**: Numerical operations
- **Albumentations**: Advanced data augmentation

### 8.2 Evaluation & Visualization
- **scikit-learn**: Metrics calculation
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical visualizations
- **tqdm**: Progress bars

## 9. Deployment Considerations

### 9.1 Offline Operation
- All models run locally without internet
- No cloud dependencies
- No database requirements
- Pretrained weights downloaded once during setup

### 9.2 CPU Optimization
- Batch size optimized for CPU (16)
- No CUDA dependencies
- Efficient inference pipeline
- Memory-conscious design

### 9.3 Portability
- Pure Python implementation
- Cross-platform compatible (Windows/Linux/Mac)
- Minimal dependencies
- Easy integration with web/desktop frontends

## 10. Limitations & Future Work

### 10.1 Current Limitations
1. **TB Stage Labels**: Currently using synthetic stage labels for demonstration
   - Production requires real stage annotations from radiologists
2. **Pneumonia Data**: Limited real pneumonia samples
   - Augmented data used as fallback
3. **Training Duration**: 10 epochs for quick testing
   - Full training (50+ epochs) recommended for production

### 10.2 Future Improvements
1. **Real Stage Annotations**: Collaborate with medical experts for accurate TB staging
2. **Larger Dataset**: Include more diverse X-ray sources
3. **Ensemble Models**: Combine multiple architectures for better accuracy
4. **Explainability**: Add Grad-CAM visualization for model interpretability
5. **Multi-GPU Support**: Optional GPU acceleration for faster training
6. **Model Compression**: Quantization for smaller model size
7. **API Development**: REST API for easy integration
8. **Web Interface**: User-friendly frontend for clinicians

## 11. Validation Strategy

### 11.1 Cross-Validation
- Stratified K-fold validation (future work)
- Ensures robust performance estimates

### 11.2 External Validation
- Test on independent datasets (future work)
- Validate generalization to different X-ray machines

### 11.3 Clinical Validation
- Comparison with radiologist diagnoses (future work)
- Inter-rater agreement analysis

## 12. Ethical Considerations

### 12.1 Medical AI Guidelines
- System is a **decision support tool**, not a replacement for medical professionals
- All predictions should be reviewed by qualified radiologists
- Not approved for clinical use without proper validation

### 12.2 Data Privacy
- No patient data stored or transmitted
- Offline operation ensures data privacy
- HIPAA/GDPR compliant deployment possible

### 12.3 Bias Mitigation
- Diverse dataset from multiple sources
- Class balancing through weighted loss
- Regular performance audits across demographics

## 13. References

### 13.1 Datasets
1. Tawsifur Rahman et al. "Tuberculosis (TB) Chest X-ray Database" - Kaggle
2. "TBX11K Simplified Dataset" - Kaggle

### 13.2 Architecture
1. He et al. "Deep Residual Learning for Image Recognition" (ResNet)
2. Russakovsky et al. "ImageNet Large Scale Visual Recognition Challenge"

### 13.3 Medical Imaging
1. WHO Guidelines on Tuberculosis Diagnosis
2. Deep Learning in Medical Imaging: A Review

---

**Document Version**: 1.0  
**Last Updated**: February 23, 2026  
**Status**: Training in Progress