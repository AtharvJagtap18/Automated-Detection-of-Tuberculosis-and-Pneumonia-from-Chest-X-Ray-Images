# Comprehensive Visualizations Guide

## Overview

The ablation study generates comprehensive visualizations for each CNN architecture to evaluate and compare their performance. This document explains all generated plots and metrics.

## Generated Visualizations

### For Each Architecture (ResNet50, DenseNet121, VGG16, MobileNetV2)

Each model gets the following visualizations in `backend/results/ablation_visualizations/{architecture}/`:

#### 1. Training Curves (`{arch}_training_curves.png`)

**Two plots side by side:**

**Left: Loss vs Epoch**
- Blue line with circles: Training Loss
- Red line with squares: Validation Loss
- Shows how loss decreases over epochs
- Gap between lines indicates overfitting (if validation loss increases)

**Right: Accuracy vs Epoch**
- Blue line with circles: Training Accuracy (%)
- Red line with squares: Validation Accuracy (%)
- Shows model learning progress
- Validation accuracy is the key metric

**What to look for:**
- ✅ Both losses decreasing: Good learning
- ✅ Small gap between train/val: Good generalization
- ⚠️ Validation loss increasing: Overfitting
- ⚠️ Both losses flat: Learning stalled

---

#### 2. Confusion Matrix (`{arch}_confusion_matrix.png`)

**Two heatmaps side by side:**

**Left: Counts**
- Shows actual number of predictions
- Rows: True labels
- Columns: Predicted labels
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications

**Right: Percentages**
- Shows proportion of predictions
- Each row sums to 100%
- Easier to compare across classes

**Example interpretation:**
```
              Predicted
           N     P     TB
True  N  [95%   3%    2%]   ← 95% of Normal correctly classified
      P  [5%   88%    7%]   ← 88% of Pneumonia correctly classified
      TB [2%    5%   93%]   ← 93% of TB correctly classified
```

**What to look for:**
- ✅ High diagonal values (>85%): Good performance
- ⚠️ High off-diagonal: Confusion between classes
- Common: Pneumonia ↔ TB confusion (similar appearance)

---

#### 3. ROC Curves (`{arch}_roc_curves.png`)

**Receiver Operating Characteristic curves for each class**

**Components:**
- Blue line: Normal class ROC
- Red line: Pneumonia class ROC
- Green line: Tuberculosis class ROC
- Pink dashed line: Micro-average ROC
- Black dashed line: Random classifier (baseline)

**AUC (Area Under Curve) scores:**
- 1.0 = Perfect classifier
- 0.9-1.0 = Excellent
- 0.8-0.9 = Good
- 0.7-0.8 = Fair
- 0.5-0.7 = Poor
- 0.5 = Random guessing

**What to look for:**
- ✅ Curves closer to top-left corner: Better performance
- ✅ AUC > 0.9: Excellent discrimination
- ⚠️ Curve near diagonal: Poor discrimination

**Saved separately:** `{arch}_auc_scores.json`
```json
{
  "micro_average": 0.92,
  "per_class": {
    "Normal": 0.95,
    "Pneumonia": 0.88,
    "Tuberculosis": 0.93
  }
}
```

---

### Comparison Visualizations (All Models)

Located in `backend/results/ablation_visualizations/`:

#### 4. Metrics Comparison (`all_models_metrics_comparison.png`)

**Four bar charts comparing all architectures:**

**Top-Left: Validation Accuracy**
- Shows best validation accuracy achieved
- Higher is better
- Expected: ResNet50 > DenseNet121 > MobileNetV2 > VGG16

**Top-Right: Precision**
- Proportion of correct positive predictions
- Important for minimizing false positives
- High precision = fewer false alarms

**Bottom-Left: Recall**
- Proportion of actual positives correctly identified
- Important for minimizing false negatives
- High recall = fewer missed cases

**Bottom-Right: F1-Score**
- Harmonic mean of precision and recall
- Balanced metric
- Best overall performance indicator

**What to look for:**
- ✅ Consistent high values across all metrics: Robust model
- ⚠️ High precision, low recall: Conservative (misses cases)
- ⚠️ Low precision, high recall: Aggressive (false alarms)

---

#### 5. Efficiency Analysis (`all_models_efficiency_analysis.png`)

**Three bar charts comparing efficiency:**

**Left: Model Parameters (Millions)**
- Total number of trainable parameters
- More parameters = more capacity but slower
- Expected order: VGG16 >> ResNet50 > DenseNet121 > MobileNetV2

**Middle: Model Size (MB)**
- Disk space required for model file
- Important for deployment
- Expected order: VGG16 >> ResNet50 > DenseNet121 > MobileNetV2

**Right: Training Time (Minutes)**
- Time to train for 10 epochs on CPU
- Expected order: VGG16 > DenseNet121 > ResNet50 > MobileNetV2

**What to look for:**
- ✅ Small size + high accuracy: Efficient model (DenseNet121, MobileNetV2)
- ⚠️ Large size + low accuracy: Inefficient (VGG16)

---

## Metrics Explained

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Overall correctness
- Can be misleading with imbalanced classes
- Target: >85% for medical imaging

### Precision
```
Precision = TP / (TP + FP)
```
- "Of all positive predictions, how many were correct?"
- Important when false positives are costly
- High precision = trustworthy positive predictions

### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
- "Of all actual positives, how many did we find?"
- Important when false negatives are costly
- High recall = few missed cases

### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Balanced metric
- Useful when you need both precision and recall
- Harmonic mean penalizes extreme values

### AUC (Area Under ROC Curve)
- Measures classifier's ability to distinguish classes
- Independent of classification threshold
- Robust to class imbalance
- Best metric for medical diagnosis

---

## Expected Results (10 Epochs)

### ResNet50
```
Validation Accuracy: 88-92%
Precision: 89%
Recall: 88%
F1-Score: 88.5%
AUC: 0.92-0.95

Parameters: 23.5M
Model Size: 90 MB
Training Time: 35-40 min
```

**Confusion Matrix:**
- Normal: 95% accuracy
- Pneumonia: 88% accuracy (some confusion with TB)
- TB: 93% accuracy

**ROC AUC:**
- Normal: 0.95
- Pneumonia: 0.88
- TB: 0.93

---

### DenseNet121
```
Validation Accuracy: 87-91%
Precision: 88%
Recall: 87%
F1-Score: 87.5%
AUC: 0.91-0.94

Parameters: 7.0M (3.4x smaller than ResNet50)
Model Size: 28 MB (3.2x smaller)
Training Time: 40-45 min
```

**Confusion Matrix:**
- Normal: 94% accuracy
- Pneumonia: 86% accuracy
- TB: 92% accuracy

**ROC AUC:**
- Normal: 0.94
- Pneumonia: 0.87
- TB: 0.92

---

### MobileNetV2
```
Validation Accuracy: 83-87%
Precision: 84%
Recall: 83%
F1-Score: 83.5%
AUC: 0.88-0.91

Parameters: 2.2M (10.7x smaller than ResNet50)
Model Size: 9 MB (10x smaller)
Training Time: 25-30 min (fastest)
```

**Confusion Matrix:**
- Normal: 92% accuracy
- Pneumonia: 82% accuracy
- TB: 89% accuracy

**ROC AUC:**
- Normal: 0.92
- Pneumonia: 0.84
- TB: 0.89

---

### VGG16
```
Validation Accuracy: 82-86%
Precision: 83%
Recall: 82%
F1-Score: 82.5%
AUC: 0.87-0.90

Parameters: 134.3M (5.7x larger than ResNet50)
Model Size: 528 MB (5.9x larger)
Training Time: 45-50 min (slowest)
```

**Confusion Matrix:**
- Normal: 90% accuracy
- Pneumonia: 78% accuracy (worst)
- TB: 86% accuracy

**ROC AUC:**
- Normal: 0.91
- Pneumonia: 0.82
- TB: 0.88

---

## How to Interpret Results

### Best Overall Performance
**Winner: ResNet50**
- Highest accuracy across all metrics
- Best confusion matrix (fewest errors)
- Highest AUC scores
- Trade-off: Moderate size and training time

### Best Efficiency
**Winner: DenseNet121**
- Nearly same accuracy as ResNet50 (1-2% difference)
- 3.4x fewer parameters
- 3.2x smaller model size
- Excellent accuracy-to-size ratio

### Best for Mobile/Edge
**Winner: MobileNetV2**
- Smallest model (9 MB)
- Fastest training and inference
- Acceptable accuracy (83-87%)
- 10x smaller than ResNet50

### Worst Choice
**Avoid: VGG16**
- Lowest accuracy
- Largest model (528 MB)
- Slowest training
- Outdated architecture

---

## File Structure

```
backend/results/ablation_visualizations/
├── resnet50/
│   ├── resnet50_training_curves.png
│   ├── resnet50_confusion_matrix.png
│   ├── resnet50_roc_curves.png
│   └── resnet50_auc_scores.json
├── densenet121/
│   ├── densenet121_training_curves.png
│   ├── densenet121_confusion_matrix.png
│   ├── densenet121_roc_curves.png
│   └── densenet121_auc_scores.json
├── vgg16/
│   ├── vgg16_training_curves.png
│   ├── vgg16_confusion_matrix.png
│   ├── vgg16_roc_curves.png
│   └── vgg16_auc_scores.json
├── mobilenetv2/
│   ├── mobilenetv2_training_curves.png
│   ├── mobilenetv2_confusion_matrix.png
│   ├── mobilenetv2_roc_curves.png
│   └── mobilenetv2_auc_scores.json
├── all_models_metrics_comparison.png
└── all_models_efficiency_analysis.png
```

---

## Using Visualizations in Your Report

### For Methodology Section
1. Include training curves to show learning process
2. Show confusion matrix to demonstrate classification performance
3. Include ROC curves to show discrimination ability

### For Results Section
1. Use metrics comparison chart for overall performance
2. Include efficiency analysis for practical considerations
3. Show per-class performance from confusion matrices

### For Discussion Section
1. Compare AUC scores across models
2. Discuss trade-offs (accuracy vs size vs speed)
3. Justify final model selection

### Recommended Figures for Paper
1. **Figure 1**: All models metrics comparison (4 subplots)
2. **Figure 2**: Best model (ResNet50) training curves
3. **Figure 3**: Best model confusion matrix
4. **Figure 4**: Best model ROC curves
5. **Figure 5**: Efficiency analysis (all models)

---

## Running the Ablation Study

```bash
cd "TB VS NEMO"
python ablation_study.py
```

**Time**: 2-3 hours (trains all 4 models)

**Output**:
- All visualizations (12 plots per model + 2 comparison plots)
- CSV file with numerical results
- JSON file with detailed metrics
- AUC scores for each model

---

## Troubleshooting

### Issue: Plots not generated
**Solution**: Install matplotlib and seaborn
```bash
pip install matplotlib seaborn
```

### Issue: Out of memory during visualization
**Solution**: Close plots after saving
```python
plt.close(fig)  # Already handled in code
```

### Issue: Low resolution plots
**Solution**: Increase DPI in visualizations.py
```python
plt.rcParams['savefig.dpi'] = 300  # Already set
```

---

**Document Version**: 1.0  
**Last Updated**: February 24, 2026  
**Status**: Ready to Use
