"""
Model evaluation module
Evaluates trained models on test datasets and generates comprehensive reports
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import json

from config import (
    DISEASE_MODEL_PATH,
    STAGE_MODEL_PATH,
    DISEASE_MODEL_ARCH,
    STAGE_MODEL_ARCH,
    DISEASE_CLASSES,
    TB_STAGE_CLASSES,
    DEVICE,
    EVAL_BATCH_SIZE,
    PLOTS_DIR,
    METRICS_DIR
)
from preprocess import XRayPreprocessor
from models_arch import load_model
from metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_class_distribution,
    save_metrics_report,
    print_metrics_summary
)


class XRayDataset(Dataset):
    """
    Simple dataset class for evaluation
    Expects a list of (image_path, label) tuples
    """
    
    def __init__(
        self,
        data: List[Tuple[str, int]],
        preprocessor: XRayPreprocessor
    ):
        """
        Initialize dataset
        
        Args:
            data: List of (image_path, label) tuples
            preprocessor: XRayPreprocessor instance
        """
        self.data = data
        self.preprocessor = preprocessor
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.data[idx]
        
        try:
            # Preprocess image
            tensor = self.preprocessor.preprocess(image_path)
            tensor = tensor.squeeze(0)  # Remove batch dimension
            return tensor, label
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {str(e)}")
            # Return dummy tensor on error
            return torch.zeros(3, 224, 224), label


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = DEVICE
) -> Tuple[List[int], List[int], List[float]]:
    """
    Evaluate model on dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for test data
        device: Device to run on
        
    Returns:
        Tuple of (true_labels, predicted_labels, confidences)
    """
    model.eval()
    model.to(device)
    
    all_true = []
    all_pred = []
    all_conf = []
    
    print(f"🔍 Evaluating model...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward pass
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            
            # Get predictions
            confidences, predictions = torch.max(probs, dim=1)
            
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())
            all_conf.extend(confidences.cpu().numpy())
    
    return all_true, all_pred, all_conf


def evaluate_disease_model(
    test_data: List[Tuple[str, int]],
    model_path: Optional[str] = None,
    save_results: bool = True
) -> dict:
    """
    Evaluate disease classification model
    
    Args:
        test_data: List of (image_path, label) tuples
        model_path: Path to model weights (optional)
        save_results: If True, save plots and reports
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Disease Classification Model")
    print(f"{'='*60}\n")
    
    # Load model
    model_path = model_path or str(DISEASE_MODEL_PATH)
    model = load_model(
        model_path=model_path,
        arch=DISEASE_MODEL_ARCH,
        num_classes=len(DISEASE_CLASSES),
        model_type="disease"
    )
    
    # Create dataset and dataloader
    preprocessor = XRayPreprocessor()
    dataset = XRayDataset(test_data, preprocessor)
    dataloader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    print(f"📊 Test set size: {len(dataset)} images")
    
    # Evaluate
    y_true, y_pred, confidences = evaluate_model(model, dataloader)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, DISEASE_CLASSES)
    metrics['mean_confidence'] = float(np.mean(confidences))
    metrics['std_confidence'] = float(np.std(confidences))
    
    # Print summary
    print_metrics_summary(metrics, "Disease Classifier")
    
    print(f"\n📈 Confidence Statistics:")
    print(f"   Mean: {metrics['mean_confidence']:.4f}")
    print(f"   Std:  {metrics['std_confidence']:.4f}")
    
    # Save results
    if save_results:
        print(f"\n💾 Saving results...")
        
        # Save confusion matrix
        plot_confusion_matrix(
            y_true,
            y_pred,
            DISEASE_CLASSES,
            save_path=PLOTS_DIR / "disease_confusion_matrix.png",
            title="Disease Classification - Confusion Matrix"
        )
        
        # Save class distribution
        plot_class_distribution(
            y_true,
            DISEASE_CLASSES,
            save_path=PLOTS_DIR / "disease_class_distribution.png",
            title="Disease Classification - Test Set Distribution"
        )
        
        # Save metrics report
        save_metrics_report(
            metrics,
            METRICS_DIR / "disease_metrics",
            "Disease Classifier"
        )
    
    return {
        'metrics': metrics,
        'predictions': {
            'true_labels': y_true,
            'predicted_labels': y_pred,
            'confidences': confidences
        }
    }


def evaluate_stage_model(
    test_data: List[Tuple[str, int]],
    model_path: Optional[str] = None,
    save_results: bool = True
) -> dict:
    """
    Evaluate TB stage classification model
    
    Args:
        test_data: List of (image_path, stage_label) tuples (TB images only)
        model_path: Path to model weights (optional)
        save_results: If True, save plots and reports
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating TB Stage Classification Model")
    print(f"{'='*60}\n")
    
    # Load model
    model_path = model_path or str(STAGE_MODEL_PATH)
    model = load_model(
        model_path=model_path,
        arch=STAGE_MODEL_ARCH,
        num_classes=len(TB_STAGE_CLASSES),
        model_type="stage"
    )
    
    # Create dataset and dataloader
    preprocessor = XRayPreprocessor()
    dataset = XRayDataset(test_data, preprocessor)
    dataloader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"📊 Test set size: {len(dataset)} TB images")
    
    # Evaluate
    y_true, y_pred, confidences = evaluate_model(model, dataloader)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, TB_STAGE_CLASSES)
    metrics['mean_confidence'] = float(np.mean(confidences))
    metrics['std_confidence'] = float(np.std(confidences))
    
    # Print summary
    print_metrics_summary(metrics, "TB Stage Classifier")
    
    print(f"\n📈 Confidence Statistics:")
    print(f"   Mean: {metrics['mean_confidence']:.4f}")
    print(f"   Std:  {metrics['std_confidence']:.4f}")
    
    # Save results
    if save_results:
        print(f"\n💾 Saving results...")
        
        # Save confusion matrix
        plot_confusion_matrix(
            y_true,
            y_pred,
            TB_STAGE_CLASSES,
            save_path=PLOTS_DIR / "stage_confusion_matrix.png",
            title="TB Stage Classification - Confusion Matrix"
        )
        
        # Save class distribution
        plot_class_distribution(
            y_true,
            TB_STAGE_CLASSES,
            save_path=PLOTS_DIR / "stage_class_distribution.png",
            title="TB Stage Classification - Test Set Distribution"
        )
        
        # Save metrics report
        save_metrics_report(
            metrics,
            METRICS_DIR / "stage_metrics",
            "TB Stage Classifier"
        )
    
    return {
        'metrics': metrics,
        'predictions': {
            'true_labels': y_true,
            'predicted_labels': y_pred,
            'confidences': confidences
        }
    }


# Example usage
if __name__ == "__main__":
    print("🧪 Testing evaluation module...")
    print("⚠️  Note: This requires actual test data")
    print("   Create test_data as: [(image_path, label), ...]")
    
    # Example structure (replace with actual data):
    # disease_test_data = [
    #     ("path/to/normal1.jpg", 0),
    #     ("path/to/pneumonia1.jpg", 1),
    #     ("path/to/tb1.jpg", 2),
    #     ...
    # ]
    # 
    # stage_test_data = [
    #     ("path/to/tb_stage1.jpg", 0),
    #     ("path/to/tb_stage2.jpg", 1),
    #     ...
    # ]
    # 
    # # Evaluate disease model
    # disease_results = evaluate_disease_model(disease_test_data)
    # 
    # # Evaluate stage model
    # stage_results = evaluate_stage_model(stage_test_data)
    
    print("\n✅ Evaluation module ready!")
    print("   Use evaluate_disease_model() and evaluate_stage_model()")
    print("   with your test data to generate comprehensive reports.")
