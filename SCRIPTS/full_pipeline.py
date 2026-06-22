
import os
import sys
from pathlib import Path
import shutil
import json
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

print("🚀 TB X-Ray Analysis - Complete Training Pipeline")
print("="*70)

# Step 1: Install dependencies
print("\n📦 Step 1: Installing dependencies...")
try:
    import torch
    import torchvision
    import kagglehub
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
    print("✅ All dependencies installed")
except ImportError as e:
    print(f"⚠️  Missing dependency: {e}")
    print("Installing required packages...")
    os.system("pip install torch torchvision kagglehub pillow numpy tqdm albumentations scikit-learn matplotlib seaborn")
    print("✅ Dependencies installed. Please run the script again.")
    sys.exit(0)

# Import backend modules
from backend.config import DISEASE_CLASSES, TB_STAGE_CLASSES
from backend.data_preparation import (
    load_dataset_from_folders,
    split_dataset,
    analyze_dataset,
    save_dataset_split
)
from backend.train_disease import train_disease_classifier
from backend.train_stage import train_stage_classifier


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print("✅ Kaggle credentials found")
        return True
    else:
        print("❌ Kaggle credentials not found")
        print("\n" + "="*70)
        print("📝 KAGGLE API SETUP REQUIRED")
        print("="*70)
        print("\nTo download datasets from Kaggle, you need API credentials:")
        print("\n1️⃣  Go to: https://www.kaggle.com/settings")
        print("2️⃣  Scroll to 'API' section")
        print("3️⃣  Click 'Create New Token'")
        print("4️⃣  Download kaggle.json file")
        print(f"\n5️⃣  Place kaggle.json in: {kaggle_dir}")
        print(f"    Full path: {kaggle_json}")
        print("\n6️⃣  Run this script again")
        print("="*70 + "\n")
        return False


def download_tb_dataset():
    """Download Tuberculosis TB Chest X-ray Dataset"""
    print("\n📥 Step 2a: Downloading TB Chest X-ray Dataset...")
    print("-"*70)
    
    try:
        # Download dataset
        print("Downloading TB dataset from Kaggle (this may take 10-15 minutes)...")
        path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Create target directory
        target_dir = Path("data/raw/tb_dataset")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy to our directory
        print(f"📂 Organizing dataset to: {target_dir}")
        source_path = Path(path)
        
        if source_path.exists():
            for item in source_path.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(source_path)
                    dest_file = target_dir / relative_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    if not dest_file.exists():
                        shutil.copy2(item, dest_file)
        
        print(f"✅ TB dataset ready at: {target_dir}")
        return target_dir
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Manual Download Instructions:")
        print("   1. Go to: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
        print("   2. Download the dataset")
        print("   3. Extract to: data/raw/tb_dataset/")
        print("   4. Run this script again")
        return None


def download_tbx11k_dataset():
    """Download TBX11K Simplified Dataset"""
    print("\n📥 Step 2b: Downloading TBX11K Simplified Dataset...")
    print("-"*70)
    
    try:
        # Download dataset
        print("Downloading TBX11K dataset from Kaggle (this may take 15-20 minutes)...")
        path = kagglehub.dataset_download("vbookshelf/tbx11k-simplified")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Create target directory
        target_dir = Path("data/raw/tbx11k")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy to our directory
        print(f"📂 Organizing dataset to: {target_dir}")
        source_path = Path(path)
        
        if source_path.exists():
            for item in source_path.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(source_path)
                    dest_file = target_dir / relative_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    if not dest_file.exists():
                        shutil.copy2(item, dest_file)
        
        print(f"✅ TBX11K dataset ready at: {target_dir}")
        return target_dir
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Manual Download Instructions:")
        print("   1. Go to: https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified")
        print("   2. Download the dataset")
        print("   3. Extract to: data/raw/tbx11k/")
        return None


def organize_dataset(tb_dir: Path, tbx11k_dir: Path = None):
    """Organize both TB datasets into disease classes"""
    print("\n🗂️  Step 3: Organizing Datasets...")
    print("-"*70)
    
    organized_dir = Path("data/organized")
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    for class_name in DISEASE_CLASSES:
        (organized_dir / class_name).mkdir(exist_ok=True)
    
    tb_count = 0
    normal_count = 0
    pneumonia_count = 0
    
    # Process TB Chest X-ray Dataset
    if tb_dir and tb_dir.exists():
        print("\n📂 Processing TB Chest X-ray Dataset...")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(list(tb_dir.rglob(f"*{ext}")))
            all_images.extend(list(tb_dir.rglob(f"*{ext.upper()}")))
        
        print(f"Found {len(all_images)} images in TB dataset")
        
        for img_path in tqdm(all_images, desc="Processing TB dataset"):
            img_name = img_path.name.lower()
            parent_name = img_path.parent.name.lower()
            
            # Determine class based on path or filename
            if 'tb' in parent_name or 'tuberculosis' in parent_name:
                dest_dir = organized_dir / "Tuberculosis"
                tb_count += 1
            elif 'normal' in parent_name or 'healthy' in parent_name:
                dest_dir = organized_dir / "Normal"
                normal_count += 1
            elif 'pneumonia' in parent_name:
                dest_dir = organized_dir / "Pneumonia"
                pneumonia_count += 1
            elif 'tb' in img_name or 'tuberculosis' in img_name:
                dest_dir = organized_dir / "Tuberculosis"
                tb_count += 1
            elif 'normal' in img_name or 'healthy' in img_name:
                dest_dir = organized_dir / "Normal"
                normal_count += 1
            elif 'pneumonia' in img_name:
                dest_dir = organized_dir / "Pneumonia"
                pneumonia_count += 1
            else:
                # Default to TB if unclear
                dest_dir = organized_dir / "Tuberculosis"
                tb_count += 1
            
            # Copy image with unique name
            dest_file = dest_dir / f"tb_dataset_{img_path.name}"
            if not dest_file.exists():
                shutil.copy2(img_path, dest_file)
    
    # Process TBX11K Dataset
    if tbx11k_dir and tbx11k_dir.exists():
        print("\n📂 Processing TBX11K Dataset...")
        
        all_images = []
        for ext in image_extensions:
            all_images.extend(list(tbx11k_dir.rglob(f"*{ext}")))
            all_images.extend(list(tbx11k_dir.rglob(f"*{ext.upper()}")))
        
        print(f"Found {len(all_images)} images in TBX11K")
        
        for img_path in tqdm(all_images, desc="Processing TBX11K"):
            img_name = img_path.name.lower()
            parent_name = img_path.parent.name.lower()
            
            # TBX11K typically has TB and Normal images
            if 'normal' in parent_name or 'healthy' in parent_name or 'normal' in img_name:
                dest_dir = organized_dir / "Normal"
                normal_count += 1
            elif 'tb' in parent_name or 'tuberculosis' in parent_name or 'tb' in img_name:
                dest_dir = organized_dir / "Tuberculosis"
                tb_count += 1
            else:
                # Default to TB (TBX11K is primarily TB dataset)
                dest_dir = organized_dir / "Tuberculosis"
                tb_count += 1
            
            # Copy image with unique name
            dest_file = dest_dir / f"tbx11k_{img_path.name}"
            if not dest_file.exists():
                shutil.copy2(img_path, dest_file)
    
    print(f"\n✅ Dataset organized:")
    print(f"   Normal:       {normal_count} images")
    print(f"   Pneumonia:    {pneumonia_count} images")
    print(f"   Tuberculosis: {tb_count} images")
    print(f"   Total:        {normal_count + pneumonia_count + tb_count} images")
    
    return organized_dir


def add_synthetic_pneumonia(organized_dir: Path, num_samples: int = 100):
    """
    Add synthetic pneumonia samples if real pneumonia data is not available
    (Only used as fallback)
    """
    pneumonia_dir = organized_dir / "Pneumonia"
    existing_pneumonia = list(pneumonia_dir.glob("*.jpg")) + list(pneumonia_dir.glob("*.jpeg")) + list(pneumonia_dir.glob("*.png"))
    
    if len(existing_pneumonia) > 100:
        print(f"\n✅ Real pneumonia data found: {len(existing_pneumonia)} images")
        print("   Skipping synthetic data generation")
        return
    
    print("\n🔬 Step 4: Adding Synthetic Pneumonia Samples (Fallback)...")
    print("-"*70)
    print("⚠️  Using augmented TB images as pneumonia placeholders")
    print("   Real pneumonia dataset is recommended!")
    
    # Get some TB images to augment
    tb_dir = organized_dir / "Tuberculosis"
    tb_images = list(tb_dir.glob("*.jpg"))[:num_samples]
    
    if not tb_images:
        tb_images = list(tb_dir.glob("*.png"))[:num_samples]
    
    print(f"Creating {len(tb_images)} synthetic pneumonia samples...")
    
    from PIL import Image, ImageEnhance, ImageFilter
    
    for i, img_path in enumerate(tqdm(tb_images, desc="Augmenting")):
        try:
            img = Image.open(img_path)
            
            # Apply augmentations to make it different
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)
            
            # Save as pneumonia
            dest_path = pneumonia_dir / f"pneumonia_synthetic_{i:04d}.jpg"
            img.save(dest_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"✅ Added {len(list(pneumonia_dir.glob('*.jpg')))} pneumonia samples")


def prepare_training_data(organized_dir: Path):
    """Prepare train/val/test splits"""
    print("\n📊 Step 5: Preparing Train/Val/Test Splits...")
    print("-"*70)
    
    # Load dataset
    data = load_dataset_from_folders(
        organized_dir,
        DISEASE_CLASSES,
        extensions=['.jpg', '.jpeg', '.png']
    )
    
    if len(data) == 0:
        print("❌ No images found!")
        return None, None, None
    
    # Analyze
    analyze_dataset(data, DISEASE_CLASSES)
    
    # Split
    train_data, val_data, test_data = split_dataset(
        data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True
    )
    
    # Save split info
    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    
    save_dataset_split(
        train_data,
        val_data,
        test_data,
        split_dir / "disease_dataset_split.json",
        DISEASE_CLASSES
    )
    
    return train_data, val_data, test_data


def train_disease_model(train_data, val_data):
    """Train disease classification model"""
    print("\n🎓 Step 6: Training Disease Classification Model...")
    print("-"*70)
    
    if not train_data or not val_data:
        print("❌ No training data available")
        return None
    
    try:
        results = train_disease_classifier(
            train_data=train_data,
            val_data=val_data,
            model_arch="resnet50",
            num_epochs=10,  # Quick training for testing
            batch_size=16,  # Smaller batch for CPU
            learning_rate=0.001,
            use_class_weights=True,
            freeze_backbone=False,
            patience=10
        )
        
        print("\n✅ Disease model training complete!")
        print(f"   Best validation accuracy: {results['best_val_acc']*100:.2f}%")
        print(f"   Training time: {results['training_time']/60:.2f} minutes")
        
        return results
        
    except Exception as e:
        print(f"❌ Error training disease model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def prepare_tb_stage_data(organized_dir: Path):
    """
    Prepare TB stage data (synthetic for demo)
    In production, you need real TB stage labels
    """
    print("\n🔬 Step 7: Preparing TB Stage Data...")
    print("-"*70)
    print("⚠️  Creating synthetic TB stage labels for demonstration")
    print("   For production, you need real TB stage annotations!")
    
    tb_dir = organized_dir / "Tuberculosis"
    tb_images = list(tb_dir.glob("*.jpg")) + list(tb_dir.glob("*.png"))
    
    if not tb_images:
        print("❌ No TB images found")
        return None, None, None
    
    # Randomly assign stages (for demo only!)
    import random
    random.seed(42)
    
    stage_data = []
    for img_path in tb_images:
        stage = random.randint(0, 4)  # Random stage 0-4
        stage_data.append((str(img_path), stage))
    
    # Split
    train_data, val_data, test_data = split_dataset(
        stage_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True
    )
    
    print(f"✅ TB stage data prepared:")
    print(f"   Training: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    
    # Save split
    split_dir = Path("TB VS NEMO/data/splits")
    save_dataset_split(
        train_data,
        val_data,
        test_data,
        split_dir / "stage_dataset_split.json",
        TB_STAGE_CLASSES
    )
    
    return train_data, val_data, test_data


def train_stage_model(train_data, val_data):
    """Train TB stage classification model"""
    print("\n🎓 Step 8: Training TB Stage Classification Model...")
    print("-"*70)
    
    if not train_data or not val_data:
        print("❌ No training data available")
        return None
    
    try:
        results = train_stage_classifier(
            train_data=train_data,
            val_data=val_data,
            model_arch="resnet50",
            num_epochs=10,
            batch_size=16,
            learning_rate=0.001,
            use_class_weights=True,
            freeze_backbone=False,
            patience=10
        )
        
        print("\n✅ TB stage model training complete!")
        print(f"   Best validation accuracy: {results['best_val_acc']*100:.2f}%")
        print(f"   Training time: {results['training_time']/60:.2f} minutes")
        
        return results
        
    except Exception as e:
        print(f"❌ Error training stage model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_inference():
    """Test the trained models"""
    print("\n🧪 Step 9: Testing Inference...")
    print("-"*70)
    
    try:
        from backend.inference import XRayPredictor
        
        predictor = XRayPredictor()
        
        # Test with a random image from dataset
        test_dir = Path("data/organized/Tuberculosis")
        test_images = list(test_dir.glob("*.jpg"))[:3]
        
        if not test_images:
            test_images = list(test_dir.glob("*.png"))[:3]
        
        if test_images:
            print(f"Testing on {len(test_images)} sample images...")
            
            for img_path in test_images:
                result = predictor.predict(img_path, return_timing=True)
                
                print(f"\n📸 Image: {img_path.name}")
                print(f"   Disease: {result['disease']}")
                print(f"   Confidence: {result['confidence']:.2%}")
                
                if 'tb_stage' in result:
                    print(f"   TB Stage: {result['tb_stage']}")
                    print(f"   Stage Confidence: {result.get('stage_confidence', 0):.2%}")
                
                if 'timing' in result:
                    print(f"   Inference time: {result['timing']['total_time']:.3f}s")
        
        print("\n✅ Inference test complete!")
        
    except Exception as e:
        print(f"❌ Error testing inference: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main pipeline"""
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("🚀 COMPLETE TB X-RAY ANALYSIS PIPELINE")
    print("="*70)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check Kaggle credentials
    if not check_kaggle_credentials():
        print("\n❌ Pipeline stopped: Kaggle credentials required")
        print("   Please setup Kaggle API and run again")
        return
    
    # Step 2: Download datasets
    print("\n📥 Downloading datasets (this will take 20-30 minutes)...")
    tb_dir = download_tb_dataset()
    tbx11k_dir = download_tbx11k_dataset()
    
    if not tb_dir and not tbx11k_dir:
        print("\n❌ Pipeline stopped: No datasets downloaded successfully")
        return
    
    # Step 3: Organize datasets
    organized_dir = organize_dataset(tb_dir, tbx11k_dir)
    
    # Step 4: Add synthetic pneumonia if needed (fallback)
    add_synthetic_pneumonia(organized_dir, num_samples=200)
    
    # Step 5: Prepare disease classification data
    train_data, val_data, test_data = prepare_training_data(organized_dir)
    
    if not train_data:
        print("\n❌ Pipeline stopped: No training data available")
        return
    
    # Step 6: Train disease model
    disease_results = train_disease_model(train_data, val_data)
    
    # Step 7: Prepare TB stage data
    stage_train, stage_val, stage_test = prepare_tb_stage_data(organized_dir)
    
    # Step 8: Train stage model
    stage_results = train_stage_model(stage_train, stage_val)
    
    # Step 9: Test inference
    test_inference()
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"Total time: {duration.total_seconds()/60:.2f} minutes")
    
    if disease_results:
        print(f"\n📊 Disease Model:")
        print(f"   Best accuracy: {disease_results['best_val_acc']*100:.2f}%")
        print(f"   Model saved: backend/models/disease_model.pth")
    
    if stage_results:
        print(f"\n📊 TB Stage Model:")
        print(f"   Best accuracy: {stage_results['best_val_acc']*100:.2f}%")
        print(f"   Model saved: backend/models/stage_model.pth")
    
    print(f"\n📁 Results saved in: backend/results/")
    print(f"\n🎉 You can now use the models for prediction!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
