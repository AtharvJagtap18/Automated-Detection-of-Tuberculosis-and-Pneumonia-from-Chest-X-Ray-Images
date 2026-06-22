"""
Training script for TB Stage Classification Model
Trains ResNet/DenseNet to classify TB stages 1-5
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import json
import time
from datetime import datetime

from config import (
    STAGE_MODEL_PATH,
    STAGE_MODEL_ARCH,
    TB_STAGE_CLASSES,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    DEVICE,
    RANDOM_SEED,
    RESULTS_DIR
)
from preprocess import XRayPreprocessor
from models_arch import TBStageClassifier, count_parameters
from metrics import calculate_metrics, plot_confusion_matrix
from train_disease import (
    TrainingDataset,
    EarlyStopping,
    calculate_class_weights,
    train_epoch,
    validate_epoch
)


# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def train_stage_classifier(
    train_data: List[Tuple[str, int]],
    val_data: List[Tuple[str, int]],
    model_arch: str = STAGE_MODEL_ARCH,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    save_path: Optional[str] = None,
    use_class_weights: bool = True,
    freeze_backbone: bool = False,
    patience: int = EARLY_STOPPING_PATIENCE
) -> Dict:
    """
    Train TB stage classification model
    
    Args:
        train_data: List of (TB_image_path, stage_label) tuples
        val_data: List of (TB_image_path, stage_label) tuples
        model_arch: Model architecture
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Path to save best model
        use_class_weights: Use class weights for imbalanced data
        freeze_backbone: Freeze backbone weights
        patience: Early stopping patience
        
    Returns:
        Dictionary with training history
    """
    print(f"\n{'='*70}")
    print(f"Training TB Stage Classification Model")
    print(f"{'='*70}\n")
    
    # Set device
    device = torch.device(DEVICE)
    print(f"🖥️  Device: {device}")
    
    # Create model
    print(f"🏗️  Building {model_arch.upper()} model...")
    model = TBStageClassifier(
        arch=model_arch,
        num_stages=len(TB_STAGE_CLASSES),
        pretrained=True,
        freeze_backbone=freeze_backbone
    )
    model.to(device)
    
    trainable, total = count_parameters(model)
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Total parameters: {total:,}")
    
    # Create datasets
    print(f"\n📊 Creating datasets...")
    preprocessor = XRayPreprocessor()
    
    train_dataset = TrainingDataset(train_data, preprocessor, training=True)
    val_dataset = TrainingDataset(val_data, preprocessor, training=False)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Calculate class weights
    if use_class_weights:
        train_labels = [label for _, label in train_data]
        class_weights = calculate_class_weights(train_labels, len(TB_STAGE_CLASSES))
        class_weights = class_weights.to(device)
        print(f"\n⚖️  Class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = None
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Training loop
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {patience}")
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_labels, val_preds = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        print(f"   Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_path or str(STAGE_MODEL_PATH)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history,
                'config': {
                    'arch': model_arch,
                    'num_stages': len(TB_STAGE_CLASSES),
                    'stage_names': TB_STAGE_CLASSES
                }
            }
            
            torch.save(checkpoint, save_path)
            print(f"   ✅ Best model saved! (Val Acc: {val_acc*100:.2f}%)")
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    # Training complete
    training_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"{'='*70}")
    print(f"   Total time: {training_time/60:.2f} minutes")
    print(f"   Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"   Model saved to: {save_path}")
    
    # Final evaluation
    print(f"\n📊 Final Validation Metrics:")
    val_loss, val_acc, val_labels, val_preds = validate_epoch(
        model, val_loader, criterion, device
    )
    
    metrics = calculate_metrics(val_labels, val_preds, TB_STAGE_CLASSES)
    
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    
    # Save training history
    history_path = RESULTS_DIR / f"stage_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n💾 Training history saved to: {history_path}")
    
    # Plot final confusion matrix
    cm_path = RESULTS_DIR / "plots" / f"stage_final_confusion_matrix.png"
    plot_confusion_matrix(
        val_labels,
        val_preds,
        TB_STAGE_CLASSES,
        save_path=cm_path,
        title="TB Stage Classifier - Final Validation"
    )
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'final_metrics': metrics,
        'training_time': training_time
    }


# Example usage
if __name__ == "__main__":
    print("🧪 TB Stage Classifier Training Script")
    print("⚠️  This requires TB images with stage labels")
    print("\nExample usage:")
    print("""
    # Prepare TB data with stages
    train_data = [
        ("path/to/tb_stage1_1.jpg", 0),  # Stage 1
        ("path/to/tb_stage2_1.jpg", 1),  # Stage 2
        ("path/to/tb_stage3_1.jpg", 2),  # Stage 3
        # ... more TB images with stages
    ]
    
    val_data = [
        # ... validation TB images
    ]
    
    # Train model
    results = train_stage_classifier(
        train_data=train_data,
        val_data=val_data,
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    """)
