"""
Ablation Study: Compare Different CNN Architectures
Tests ResNet50, DenseNet121, VGG16, and MobileNetV2 for TB X-ray classification
Generates comprehensive visualizations: confusion matrix, ROC curves, training curves, metrics
"""
import sys
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backend.train_disease import train_disease_classifier, TrainingDataset, validate_epoch
from backend.data_preparation import load_dataset_from_folders, split_dataset
from backend.config import DISEASE_CLASSES
from backend.preprocess import XRayPreprocessor
from backend.models_arch import DiseaseClassifier, count_parameters
from backend.visualizations import (
    create_comprehensive_report,
    plot_metrics_comparison,
    plot_efficiency_analysis
)


# Architectures to compare
ARCHITECTURES = [
    "resnet50",
    "densenet121", 
    "vgg16",
    "mobilenetv2"
]

# Training configuration
EPOCHS = 10  # Quick comparison
BATCH_SIZE = 16
LEARNING_RATE = 0.001


def load_prepared_data():
    """Load already organized dataset"""
    print("\n📊 Loading prepared dataset...")
    
    organized_dir = Path("data/organized")
    
    if not organized_dir.exists():
        print("❌ Dataset not found. Please run full_pipeline.py first!")
        return None, None, None
    
    # Load dataset
    data = load_dataset_from_folders(
        organized_dir,
        DISEASE_CLASSES,
        extensions=['.jpg', '.jpeg', '.png']
    )
    
    if len(data) == 0:
        print("❌ No images found!")
        return None, None, None
    
    print(f"✅ Loaded {len(data)} images")
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(
        data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True
    )
    
    print(f"   Training: {len(train_data)}")
    print(f"   Validation: {len(val_data)}")
    print(f"   Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def get_predictions_and_probs(model, dataloader, device):
    """Get predictions and probabilities for visualization"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
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


def train_single_architecture(arch, train_data, val_data, epochs=10):
    """Train a single architecture and return results with visualizations"""
    print(f"\n{'='*70}")
    print(f"🏗️  Training {arch.upper()}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Train model
        results = train_disease_classifier(
            train_data=train_data,
            val_data=val_data,
            model_arch=arch,
            num_epochs=epochs,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            save_path=f"backend/models/ablation_{arch}_disease.pth",
            use_class_weights=True,
            freeze_backbone=False,
            patience=5
        )
        
        training_time = time.time() - start_time
        
        # Load best model for evaluation
        device = torch.device('cpu')
        model = DiseaseClassifier(arch=arch, num_classes=len(DISEASE_CLASSES), pretrained=False)
        checkpoint = torch.load(f"backend/models/ablation_{arch}_disease.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Get predictions and probabilities for visualizations
        print(f"\n📊 Generating predictions for visualizations...")
        preprocessor = XRayPreprocessor()
        val_dataset = TrainingDataset(val_data, preprocessor, training=False)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        y_true, y_pred, y_probs = get_predictions_and_probs(model, val_loader, device)
        
        # Create comprehensive visualizations
        viz_dir = Path(f"backend/results/ablation_visualizations/{arch}")
        create_comprehensive_report(
            arch=arch,
            history=results['history'],
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            class_names=DISEASE_CLASSES,
            metrics=results['final_metrics'],
            save_dir=viz_dir
        )
        
        # Extract metrics
        best_val_acc = results['best_val_acc']
        final_metrics = results['final_metrics']
        
        # Count parameters
        trainable_params, total_params = count_parameters(model)
        
        # Calculate model size
        model_path = Path(f"backend/models/ablation_{arch}_disease.pth")
        model_size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
        
        return {
            'architecture': arch.upper(),
            'best_val_accuracy': best_val_acc * 100,
            'final_accuracy': final_metrics['accuracy'] * 100,
            'precision': final_metrics['precision'] * 100,
            'recall': final_metrics['recall'] * 100,
            'f1_score': final_metrics['f1_score'] * 100,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'model_size_mb': model_size_mb,
            'training_time_min': training_time / 60,
            'status': 'Success'
        }
        
    except Exception as e:
        print(f"❌ Error training {arch}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'architecture': arch.upper(),
            'best_val_accuracy': 0,
            'final_accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'trainable_params': 0,
            'total_params': 0,
            'model_size_mb': 0,
            'training_time_min': 0,
            'status': f'Failed: {str(e)}'
        }


def run_ablation_study():
    """Run complete ablation study"""
    print("\n" + "="*70)
    print("🔬 ABLATION STUDY: CNN Architecture Comparison")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nArchitectures to test: {', '.join([a.upper() for a in ARCHITECTURES])}")
    print(f"Epochs per model: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    # Load data once
    train_data, val_data, test_data = load_prepared_data()
    
    if train_data is None:
        print("\n❌ Ablation study stopped: No data available")
        return
    
    # Train each architecture
    results = []
    
    for i, arch in enumerate(ARCHITECTURES, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Model {i}/{len(ARCHITECTURES)}: {arch.upper()}")
        print(f"{'#'*70}")
        
        result = train_single_architecture(arch, train_data, val_data, EPOCHS)
        results.append(result)
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv("backend/results/ablation_study_intermediate.csv", index=False)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Sort by validation accuracy
    df = df.sort_values('best_val_accuracy', ascending=False)
    
    # Save results
    results_dir = Path("backend/results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = results_dir / f"ablation_study_{timestamp}.csv"
    json_path = results_dir / f"ablation_study_{timestamp}.json"
    
    df.to_csv(csv_path, index=False)
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n\n" + "="*70)
    print("📊 ABLATION STUDY RESULTS")
    print("="*70)
    print("\n" + df.to_string(index=False))
    
    print("\n\n" + "="*70)
    print("🏆 RANKING BY VALIDATION ACCURACY")
    print("="*70)
    
    for i, row in df.iterrows():
        print(f"\n{i+1}. {row['architecture']}")
        print(f"   Validation Accuracy: {row['best_val_accuracy']:.2f}%")
        print(f"   F1-Score: {row['f1_score']:.2f}%")
        print(f"   Parameters: {row['total_params']:,}")
        print(f"   Model Size: {row['model_size_mb']:.2f} MB")
        print(f"   Training Time: {row['training_time_min']:.2f} min")
    
    # Best model
    best = df.iloc[0]
    print("\n\n" + "="*70)
    print("🥇 BEST MODEL")
    print("="*70)
    print(f"Architecture: {best['architecture']}")
    print(f"Validation Accuracy: {best['best_val_accuracy']:.2f}%")
    print(f"Precision: {best['precision']:.2f}%")
    print(f"Recall: {best['recall']:.2f}%")
    print(f"F1-Score: {best['f1_score']:.2f}%")
    print(f"Parameters: {best['total_params']:,}")
    print(f"Model Size: {best['model_size_mb']:.2f} MB")
    print(f"Training Time: {best['training_time_min']:.2f} minutes")
    
    # Efficiency analysis
    print("\n\n" + "="*70)
    print("⚡ EFFICIENCY ANALYSIS")
    print("="*70)
    
    # Accuracy per parameter
    df['acc_per_million_params'] = df['best_val_accuracy'] / (df['total_params'] / 1e6)
    
    # Accuracy per MB
    df['acc_per_mb'] = df['best_val_accuracy'] / df['model_size_mb']
    
    # Accuracy per minute
    df['acc_per_min'] = df['best_val_accuracy'] / df['training_time_min']
    
    print("\nMost Parameter-Efficient:")
    efficient = df.nlargest(1, 'acc_per_million_params').iloc[0]
    print(f"   {efficient['architecture']}: {efficient['acc_per_million_params']:.2f} acc/M params")
    
    print("\nSmallest Model:")
    smallest = df.nsmallest(1, 'model_size_mb').iloc[0]
    print(f"   {smallest['architecture']}: {smallest['model_size_mb']:.2f} MB")
    
    print("\nFastest Training:")
    fastest = df.nsmallest(1, 'training_time_min').iloc[0]
    print(f"   {fastest['architecture']}: {fastest['training_time_min']:.2f} minutes")
    
    # Save final results
    print(f"\n\n💾 Results saved:")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {json_path}")
    
    # Create comparison visualizations
    print(f"\n📊 Creating comparison visualizations...")
    comparison_dir = Path("backend/results/ablation_visualizations")
    
    # Metrics comparison
    fig_metrics = plot_metrics_comparison(
        results,
        save_path=comparison_dir / "all_models_metrics_comparison.png",
        title="CNN Architecture Comparison - Performance Metrics"
    )
    plt.close(fig_metrics)
    
    # Efficiency analysis
    fig_efficiency = plot_efficiency_analysis(
        results,
        save_path=comparison_dir / "all_models_efficiency_analysis.png",
        title="CNN Architecture Comparison - Efficiency Metrics"
    )
    plt.close(fig_efficiency)
    
    print(f"✅ Comparison visualizations saved to: {comparison_dir}")
    
    print("\n" + "="*70)
    print("✅ ABLATION STUDY COMPLETE!")
    print("="*70)
    print(f"\n📁 All results and visualizations saved in:")
    print(f"   {Path('backend/results/ablation_visualizations').absolute()}")
    print("="*70 + "\n")
    
    return df


if __name__ == "__main__":
    try:
        results_df = run_ablation_study()
    except KeyboardInterrupt:
        print("\n\n⚠️  Ablation study interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Ablation study failed: {str(e)}")
        import traceback
        traceback.print_exc()
