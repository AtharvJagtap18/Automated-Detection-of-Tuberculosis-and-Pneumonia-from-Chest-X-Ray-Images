"""
Metrics calculation and visualization utilities
Includes confusion matrix, accuracy, precision, recall, F1-score
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from pathlib import Path
from typing import List, Dict, Optional
import json

from config import PLOTS_DIR, METRICS_DIR, CONFUSION_MATRIX_FIGSIZE, CONFUSION_MATRIX_DPI


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    average: str = 'weighted'
) -> Dict:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Per-class metrics
    precision_per_class = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    f1_per_class = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics['per_class'] = {}
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i])
        }
    
    return metrics


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Optional[Path] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot (optional)
        normalize: If True, normalize confusion matrix
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=CONFUSION_MATRIX_FIGSIZE)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=CONFUSION_MATRIX_DPI, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")
    
    return fig


def plot_class_distribution(
    y_true: List[int],
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "Class Distribution"
) -> plt.Figure:
    """
    Plot distribution of classes in dataset
    
    Args:
        y_true: True labels
        class_names: List of class names
        save_path: Path to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Count occurrences
    unique, counts = np.unique(y_true, return_counts=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart
    bars = ax.bar(range(len(unique)), counts, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=CONFUSION_MATRIX_DPI, bbox_inches='tight')
        print(f"✅ Class distribution saved to {save_path}")
    
    return fig


def save_metrics_report(
    metrics: Dict,
    save_path: Path,
    model_name: str = "Model"
):
    """
    Save metrics to JSON and text files
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save report (without extension)
        model_name: Name of the model
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = save_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {json_path}")
    
    # Save as text report
    txt_path = save_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"{model_name} - Evaluation Report\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n\n")
        
        f.write(f"Per-Class Metrics:\n")
        f.write(f"{'-'*60}\n")
        for class_name, class_metrics in metrics['per_class'].items():
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {class_metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {class_metrics['f1_score']:.4f}\n")
    
    print(f"✅ Report saved to {txt_path}")


def print_metrics_summary(metrics: Dict, model_name: str = "Model"):
    """
    Print metrics summary to console
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - Evaluation Summary")
    print(f"{'='*60}")
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\n📈 Per-Class Metrics:")
    print(f"   {'-'*56}")
    print(f"   {'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"   {'-'*56}")
    
    for class_name, class_metrics in metrics['per_class'].items():
        print(
            f"   {class_name:<20} "
            f"{class_metrics['precision']:<12.4f} "
            f"{class_metrics['recall']:<12.4f} "
            f"{class_metrics['f1_score']:<12.4f}"
        )
    
    print(f"   {'-'*56}\n")


# Example usage
if __name__ == "__main__":
    print("🧪 Testing metrics module...")
    
    # Generate dummy predictions
    np.random.seed(42)
    n_samples = 100
    n_classes = 3
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(n_samples, size=20, replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, 20)
    
    class_names = ["Normal", "Pneumonia", "Tuberculosis"]
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, class_names)
    print_metrics_summary(metrics, "Test Model")
    
    # Plot confusion matrix
    print("\n📈 Plotting confusion matrix...")
    fig = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        save_path=PLOTS_DIR / "test_confusion_matrix.png",
        title="Test Confusion Matrix"
    )
    plt.close(fig)
    
    # Plot class distribution
    print("\n📊 Plotting class distribution...")
    fig = plot_class_distribution(
        y_true,
        class_names,
        save_path=PLOTS_DIR / "test_class_distribution.png"
    )
    plt.close(fig)
    
    # Save report
    print("\n💾 Saving metrics report...")
    save_metrics_report(
        metrics,
        METRICS_DIR / "test_metrics",
        "Test Model"
    )
    
    print("\n✅ All tests passed!")
