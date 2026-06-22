"""
Create TB Severity Classification Model
Uses image-based features to classify TB severity: Mild, Moderate, Severe
This is more realistic than random staging
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json

sys.path.append(str(Path(__file__).parent / "backend"))

from backend.config import TB_STAGE_CLASSES
from backend.data_preparation import split_dataset, save_dataset_split


def analyze_tb_image_features(image_path):
    """
    Analyze TB X-ray image features to estimate severity
    Based on image characteristics (not random)
    """
    try:
        img = Image.open(image_path).convert('L')  # Grayscale
        img_array = np.array(img)
        
        # Calculate features that correlate with TB severity
        # 1. Mean intensity (darker = more infiltration)
        mean_intensity = np.mean(img_array)
        
        # 2. Standard deviation (higher = more heterogeneous = more severe)
        std_intensity = np.std(img_array)
        
        # 3. Intensity range
        intensity_range = np.max(img_array) - np.min(img_array)
        
        # 4. Dark pixel ratio (potential infiltrates)
        dark_threshold = 100
        dark_ratio = np.sum(img_array < dark_threshold) / img_array.size
        
        # Create severity score (0-1)
        # Normalize features
        mean_score = 1 - (mean_intensity / 255)  # Lower intensity = higher score
        std_score = std_intensity / 128  # Higher std = higher score
        range_score = intensity_range / 255
        dark_score = dark_ratio
        
        # Weighted severity score
        severity_score = (
            0.3 * mean_score +
            0.3 * std_score +
            0.2 * range_score +
            0.2 * dark_score
        )
        
        return severity_score
        
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return 0.5  # Default to moderate


def classify_tb_severity(severity_score):
    """
    Classify TB into 3 severity levels based on score
    More realistic than 5 random stages
    """
    if severity_score < 0.33:
        return 0, "Mild TB"
    elif severity_score < 0.67:
        return 1, "Moderate TB"
    else:
        return 2, "Severe TB"


def create_tb_severity_dataset():
    """
    Create TB severity dataset based on image features
    """
    print("\n" + "="*70)
    print("🔬 CREATING TB SEVERITY CLASSIFICATION DATASET")
    print("="*70)
    
    # Load TB images
    tb_dir = Path("data/organized/Tuberculosis")
    
    if not tb_dir.exists():
        print("❌ TB images not found. Please run full_pipeline.py first!")
        return
    
    # Get all TB images
    tb_images = list(tb_dir.glob("*.jpg")) + list(tb_dir.glob("*.png")) + list(tb_dir.glob("*.jpeg"))
    
    print(f"\n📂 Found {len(tb_images)} TB images")
    print("\n🔍 Analyzing image features for severity classification...")
    
    # Analyze each image
    severity_data = []
    severity_counts = {0: 0, 1: 0, 2: 0}
    
    from tqdm import tqdm
    for img_path in tqdm(tb_images, desc="Analyzing TB images"):
        severity_score = analyze_tb_image_features(img_path)
        severity_class, severity_name = classify_tb_severity(severity_score)
        
        severity_data.append((str(img_path), severity_class))
        severity_counts[severity_class] += 1
    
    print(f"\n✅ TB Severity Distribution:")
    print(f"   Mild TB:     {severity_counts[0]} images ({severity_counts[0]/len(tb_images)*100:.1f}%)")
    print(f"   Moderate TB: {severity_counts[1]} images ({severity_counts[1]/len(tb_images)*100:.1f}%)")
    print(f"   Severe TB:   {severity_counts[2]} images ({severity_counts[2]/len(tb_images)*100:.1f}%)")
    
    # Split dataset
    print("\n📊 Creating train/val/test splits...")
    train_data, val_data, test_data = split_dataset(
        severity_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True
    )
    
    print(f"   Training:   {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    print(f"   Test:       {len(test_data)} samples")
    
    # Save dataset
    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    
    severity_classes = ["Mild TB", "Moderate TB", "Severe TB"]
    
    save_dataset_split(
        train_data,
        val_data,
        test_data,
        split_dir / "tb_severity_dataset_split.json",
        severity_classes
    )
    
    print(f"\n💾 Dataset saved to: {split_dir / 'tb_severity_dataset_split.json'}")
    
    # Save methodology
    methodology = {
        "approach": "Image-based TB Severity Classification",
        "classes": severity_classes,
        "num_classes": 3,
        "features_used": [
            "Mean intensity (infiltration indicator)",
            "Standard deviation (heterogeneity)",
            "Intensity range (lesion contrast)",
            "Dark pixel ratio (affected area)"
        ],
        "severity_criteria": {
            "Mild TB": "Severity score < 0.33 (minimal infiltration, higher brightness)",
            "Moderate TB": "Severity score 0.33-0.67 (moderate infiltration)",
            "Severe TB": "Severity score > 0.67 (extensive infiltration, lower brightness)"
        },
        "note": "This is a proxy-based classification using image features. Clinical validation required.",
        "total_images": len(tb_images),
        "distribution": {
            "Mild TB": severity_counts[0],
            "Moderate TB": severity_counts[1],
            "Severe TB": severity_counts[2]
        }
    }
    
    with open(split_dir / "tb_severity_methodology.json", 'w') as f:
        json.dump(methodology, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ TB SEVERITY DATASET CREATED!")
    print("="*70)
    print("\n📝 Note: This uses image-based features as a proxy for severity.")
    print("   For clinical use, expert radiologist annotations are required.")
    print("\n🎯 Next steps:")
    print("   1. Train severity model: python train_tb_severity.py")
    print("   2. Generate visualizations")
    print("="*70 + "\n")
    
    return train_data, val_data, test_data, severity_classes


if __name__ == "__main__":
    try:
        create_tb_severity_dataset()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
