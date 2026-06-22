"""
Advanced Visualization Module for Model Evaluation
Includes ROC curves, AUC, training curves, and comprehensive plots
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix
)
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_training_curves(
    history: Dict,
    save_path: Optional[Path] = None,
    title: str = "Training History"
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{title} - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-o', 
             label='Training Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-s', 
             label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{title} - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"✅ Training curves saved to {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "ROC Curves"
) -> Tuple[plt.Figure, Dict]:
    """
    Plot ROC curves for multi-class classification
    
    Args:
        y_true: True labels (one-hot encoded or integer)
        y_probs: Predicted probabilities (N, num_classes)
        class_names: List of class names
        save_path: Path to save plot
        title: Plot title
        
    Returns:
        Tuple of (figure, auc_scores_dict)
    """
    n_classes = len(class_names)
    
    # Convert y_true to one-hot if needed
    if y_true.ndim == 1:
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
    else:
        y_true_onehot = y_true
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each class
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, color in zip(range(n_classes), colors[:n_classes]):
        ax.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )
    
    # Plot micro-average ROC curve
    ax.plot(
        fpr["micro"], tpr["micro"],
        label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
        color='deeppink', linestyle='--', linewidth=2
    )
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"✅ ROC curves saved to {save_path}")
    
    # Prepare AUC scores dict
    auc_scores = {
        'micro_average': float(roc_auc["micro"]),
        'per_class': {class_names[i]: float(roc_auc[i]) for i in range(n_classes)}
    }
    
    return fig, auc_scores


def plot_confusion_matrix_enhanced(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Enhanced confusion matrix with both counts and percentages
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
        cbar_kws={'label': 'Count'}
    )
    ax1.set_title(f'{title} - Counts', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Plot percentages
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
        cbar_kws={'label': 'Proportion'}
    )
    ax2.set_title(f'{title} - Percentages', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"✅ Enhanced confusion matrix saved to {save_path}")
    
    return fig


def plot_metrics_comparison(
    results: List[Dict],
    save_path: Optional[Path] = None,
    title: str = "Model Comparison"
) -> plt.Figure:
    """
    Plot comparison of multiple models
    
    Args:
        results: List of result dictionaries with 'architecture' and metrics
        save_path: Path to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    architectures = [r['architecture'] for r in results]
    metrics_to_plot = [
        ('best_val_accuracy', 'Validation Accuracy (%)', axes[0, 0]),
        ('precision', 'Precision (%)', axes[0, 1]),
        ('recall', 'Recall (%)', axes[1, 0]),
        ('f1_score', 'F1-Score (%)', axes[1, 1])
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for metric_key, metric_label, ax in metrics_to_plot:
        values = [r[metric_key] for r in results]
        bars = ax.bar(architectures, values, color=colors[:len(architectures)], alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}%',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(metric_label, fontsize=13, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(architectures, rotation=45, ha='right')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"✅ Metrics comparison saved to {save_path}")
    
    return fig


def plot_efficiency_analysis(
    results: List[Dict],
    save_path: Optional[Path] = None,
    title: str = "Efficiency Analysis"
) -> plt.Figure:
    """
    Plot efficiency metrics (parameters, size, time)
    
    Args:
        results: List of result dictionaries
        save_path: Path to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    architectures = [r['architecture'] for r in results]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Parameters
    params = [r['total_params'] / 1e6 for r in results]  # In millions
    bars1 = axes[0].bar(architectures, params, color=colors[:len(architectures)], alpha=0.7)
    axes[0].set_ylabel('Parameters (Millions)', fontsize=12)
    axes[0].set_title('Model Parameters', fontsize=13, fontweight='bold')
    axes[0].set_xticklabels(architectures, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}M', ha='center', va='bottom', fontsize=10)
    
    # Model size
    sizes = [r['model_size_mb'] for r in results]
    bars2 = axes[1].bar(architectures, sizes, color=colors[:len(architectures)], alpha=0.7)
    axes[1].set_ylabel('Model Size (MB)', fontsize=12)
    axes[1].set_title('Model Size', fontsize=13, fontweight='bold')
    axes[1].set_xticklabels(architectures, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}MB', ha='center', va='bottom', fontsize=10)
    
    # Training time
    times = [r['training_time_min'] for r in results]
    bars3 = axes[2].bar(architectures, times, color=colors[:len(architectures)], alpha=0.7)
    axes[2].set_ylabel('Training Time (Minutes)', fontsize=12)
    axes[2].set_title('Training Time', fontsize=13, fontweight='bold')
    axes[2].set_xticklabels(architectures, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}min', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"✅ Efficiency analysis saved to {save_path}")
    
    return fig


def create_comprehensive_report(
    arch: str,
    history: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    metrics: Dict,
    save_dir: Path
):
    """
    Create comprehensive visualization report for a single model
    
    Args:
        arch: Architecture name
        history: Training history
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities
        class_names: List of class names
        metrics: Metrics dictionary
        save_dir: Directory to save all plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Creating comprehensive report for {arch.upper()}...")
    
    # 1. Training curves
    fig1 = plot_training_curves(
        history,
        save_path=save_dir / f"{arch}_training_curves.png",
        title=f"{arch.upper()}"
    )
    plt.close(fig1)
    
    # 2. Confusion matrix
    fig2 = plot_confusion_matrix_enhanced(
        y_true,
        y_pred,
        class_names,
        save_path=save_dir / f"{arch}_confusion_matrix.png",
        title=f"{arch.upper()} Confusion Matrix"
    )
    plt.close(fig2)
    
    # 3. ROC curves
    fig3, auc_scores = plot_roc_curves(
        y_true,
        y_probs,
        class_names,
        save_path=save_dir / f"{arch}_roc_curves.png",
        title=f"{arch.upper()} ROC Curves"
    )
    plt.close(fig3)
    
    # Save AUC scores
    with open(save_dir / f"{arch}_auc_scores.json", 'w') as f:
        json.dump(auc_scores, f, indent=2)
    
    print(f"✅ Comprehensive report created for {arch.upper()}")
    print(f"   Saved to: {save_dir}")


# Example usage
if __name__ == "__main__":
    print("🧪 Testing visualization module...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 100
    n_classes = 3
    n_epochs = 10
    
    # Training history
    history = {
        'train_loss': np.linspace(1.0, 0.3, n_epochs) + np.random.randn(n_epochs) * 0.05,
        'val_loss': np.linspace(1.0, 0.4, n_epochs) + np.random.randn(n_epochs) * 0.08,
        'train_acc': np.linspace(0.5, 0.9, n_epochs) + np.random.randn(n_epochs) * 0.02,
        'val_acc': np.linspace(0.5, 0.85, n_epochs) + np.random.randn(n_epochs) * 0.03
    }
    
    # Predictions
    y_true = np.random.randint(0, n_classes, n_samples)
    y_probs = np.random.dirichlet(np.ones(n_classes), n_samples)
    y_pred = np.argmax(y_probs, axis=1)
    
    class_names = ["Normal", "Pneumonia", "Tuberculosis"]
    
    # Test plots
    save_dir = Path("backend/results/test_visualizations")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Testing training curves...")
    fig1 = plot_training_curves(history, save_dir / "test_training_curves.png")
    plt.close(fig1)
    
    print("\n2. Testing ROC curves...")
    fig2, auc_scores = plot_roc_curves(y_true, y_probs, class_names, 
                                       save_dir / "test_roc_curves.png")
    plt.close(fig2)
    print(f"   AUC scores: {auc_scores}")
    
    print("\n3. Testing enhanced confusion matrix...")
    fig3 = plot_confusion_matrix_enhanced(y_true, y_pred, class_names,
                                         save_dir / "test_confusion_matrix.png")
    plt.close(fig3)
    
    print("\n✅ All visualization tests passed!")
