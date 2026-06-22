"""
Generate Visualizations for ResNet50 Model
Creates confusion matrix, ROC curves, training curves, and classification report
"""
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backend.config import DISEASE_CLASSES
from backend.preprocess import XRayPreprocessor
from backend.models_arch import DiseaseClassifier
from backend.data_preparation import load_dataset_from_folders, split_dataset
from backend.train_disease import TrainingDataset
from backend.visualizations import (
    plot_training_curves,
    plot_roc_curves,
    plot_confusion_matrix_enhanced
)
from backend.metrics import calculate_metrics, print_metrics_summary
from sklearn.metrics import classification_report


def get_predictions_and_probs(model, dataloader, device):
    """Get predictions and probabilities"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Getting predictions...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def generate_visualizations():
    """Generate all visualizations for ResNet50 model"""
    
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS FOR RESNET50")
    print("="*70)
    
    # Check if model exists
    model_path = Path("backend/models/disease_model.pth")
    if not model_path.exists():
        print("\n❌ Model not found at backend/models/disease_model.pth")
        print("   Please train the model first using: python full_pipeline.py")
        return
    
    # Load model
    print("\n🔄 Loading trained ResNet50 model...")
    device = torch.device('cpu')
    model = DiseaseClassifier(arch="resnet50", num_classes=len(DISEASE_CLASSES), pretrained=False)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully")
        
        # Get training history if available
        history = checkpoint.get('history', None)
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Load dataset
    print("\n📂 Loading dataset...")
    organized_dir = Path("data/organized")
    
    if not organized_dir.exists():
        print("❌ Dataset not found. Please run full_pipeline.py first!")
        return
    
    data = load_dataset_from_folders(
        organized_dir,
        DISEASE_CLASSES,
        extensions=['.jpg', '.jpeg', '.png']
    )
    
    print(f"✅ Loaded {len(data)} images")
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(
        data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True
    )
    
    print(f"   Validation: {len(val_data)} samples")
    
    # Create dataloader
    print("\n🔄 Creating dataloader...")
    preprocessor = XRayPreprocessor()
    val_dataset = TrainingDataset(val_data, preprocessor, training=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Get predictions
    print("\n🔮 Getting predictions...")
    y_true, y_pred, y_probs = get_predictions_and_probs(model, val_loader, device)
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, DISEASE_CLASSES)
    
    # Print metrics summary
    print_metrics_summary(metrics, "ResNet50")
    
    # Print classification report
    print("\n" + "="*70)
    print("📋 CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=DISEASE_CLASSES, digits=4))
    
    # Create output directory
    output_dir = Path("backend/results/resnet50_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Generating visualizations...")
    
    # 1. Training curves (if history available)
    if history:
        print("\n1️⃣ Creating training curves...")
        fig1 = plot_training_curves(
            history,
            save_path=output_dir / "resnet50_training_curves.png",
            title="ResNet50 Training History"
        )
        plt.close(fig1)
    else:
        print("\n⚠️  Training history not available in checkpoint")
    
    # 2. Confusion matrix
    print("\n2️⃣ Creating confusion matrix...")
    fig2 = plot_confusion_matrix_enhanced(
        y_true,
        y_pred,
        DISEASE_CLASSES,
        save_path=output_dir / "resnet50_confusion_matrix.png",
        title="ResNet50 Confusion Matrix"
    )
    plt.close(fig2)
    
    # 3. ROC curves
    print("\n3️⃣ Creating ROC curves...")
    fig3, auc_scores = plot_roc_curves(
        y_true,
        y_probs,
        DISEASE_CLASSES,
        save_path=output_dir / "resnet50_roc_curves.png",
        title="ResNet50 ROC Curves"
    )
    plt.close(fig3)
    
    # Save AUC scores
    with open(output_dir / "resnet50_auc_scores.json", 'w') as f:
        json.dump(auc_scores, f, indent=2)
    
    print(f"\n   AUC Scores:")
    print(f"   Micro-average: {auc_scores['micro_average']:.4f}")
    for class_name, score in auc_scores['per_class'].items():
        print(f"   {class_name}: {score:.4f}")
    
    # 4. Save metrics to JSON
    print("\n4️⃣ Saving metrics...")
    metrics_output = {
        'model': 'ResNet50',
        'architecture': 'resnet50',
        'overall_metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score'])
        },
        'per_class_metrics': metrics['per_class'],
        'auc_scores': auc_scores
    }
    
    with open(output_dir / "resnet50_metrics.json", 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    # 5. Save classification report
    print("\n5️⃣ Saving classification report...")
    report_text = classification_report(y_true, y_pred, target_names=DISEASE_CLASSES, digits=4)
    with open(output_dir / "resnet50_classification_report.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("ResNet50 Classification Report\n")
        f.write("="*70 + "\n\n")
        f.write(report_text)
        f.write("\n\n" + "="*70 + "\n")
        f.write("Overall Metrics:\n")
        f.write("="*70 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("AUC Scores:\n")
        f.write("="*70 + "\n")
        f.write(f"Micro-average: {auc_scores['micro_average']:.4f}\n")
        for class_name, score in auc_scores['per_class'].items():
            f.write(f"{class_name}: {score:.4f}\n")
    
    # Summary
    print("\n" + "="*70)
    print("✅ VISUALIZATION GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 All files saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("   1. resnet50_training_curves.png - Loss & Accuracy vs Epoch")
    print("   2. resnet50_confusion_matrix.png - Confusion Matrix")
    print("   3. resnet50_roc_curves.png - ROC Curves with AUC")
    print("   4. resnet50_auc_scores.json - AUC scores")
    print("   5. resnet50_metrics.json - All metrics")
    print("   6. resnet50_classification_report.txt - Detailed report")
    print("\n" + "="*70)
    print("\n📊 Summary:")
    print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    print(f"   AUC (micro): {auc_scores['micro_average']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        generate_visualizations()
    except KeyboardInterrupt:
        print("\n\n⚠️  Visualization generation interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
