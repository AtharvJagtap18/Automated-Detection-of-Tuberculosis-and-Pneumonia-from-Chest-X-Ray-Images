"""
Train TB Severity Classification Model
Trains on Mild/Moderate/Severe TB classification
"""
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent / "backend"))

from backend.train_stage import train_stage_classifier


def train_severity_model():
    """Train TB severity classification model"""
    
    print("\n" + "="*70)
    print("🎓 TRAINING TB SEVERITY CLASSIFICATION MODEL")
    print("="*70)
    
    # Load dataset
    split_file = Path("data/splits/tb_severity_dataset_split.json")
    
    if not split_file.exists():
        print("\n❌ Severity dataset not found!")
        print("   Please run: python create_tb_severity_model.py")
        return
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    # Convert to format expected by training function
    train_data = [(item['path'], item['label']) for item in split_data['train']]
    val_data = [(item['path'], item['label']) for item in split_data['val']]
    
    print(f"\n📊 Dataset loaded:")
    print(f"   Training:   {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    print(f"   Classes: Mild TB, Moderate TB, Severe TB")
    
    # Train model
    results = train_stage_classifier(
        train_data=train_data,
        val_data=val_data,
        model_arch="resnet50",
        num_epochs=10,
        batch_size=16,
        learning_rate=0.001,
        save_path="backend/models/tb_severity_model.pth",
        use_class_weights=True,
        freeze_backbone=False,
        patience=5
    )
    
    print("\n" + "="*70)
    print("✅ TB SEVERITY MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Results:")
    print(f"   Best validation accuracy: {results['best_val_acc']*100:.2f}%")
    print(f"   Training time: {results['training_time']/60:.2f} minutes")
    print(f"\n💾 Model saved to: backend/models/tb_severity_model.pth")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        train_severity_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
