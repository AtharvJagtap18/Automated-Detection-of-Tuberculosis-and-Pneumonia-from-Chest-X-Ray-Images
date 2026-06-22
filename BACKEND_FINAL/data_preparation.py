"""
Data preparation utilities for training
Handles dataset loading, splitting, and organization
"""
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import random
from collections import Counter
import numpy as np
from tqdm import tqdm

from config import RANDOM_SEED, DISEASE_CLASSES, TB_STAGE_CLASSES


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_dataset_from_folders(
    data_dir: Path,
    class_folders: List[str],
    extensions: List[str] = ['.jpg', '.jpeg', '.png']
) -> List[Tuple[str, int]]:
    """
    Load dataset from folder structure:
    data_dir/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
    
    Args:
        data_dir: Root directory containing class folders
        class_folders: List of class folder names (in order)
        extensions: Valid image extensions
        
    Returns:
        List of (image_path, label) tuples
    """
    data = []
    data_dir = Path(data_dir)
    
    print(f"📂 Loading dataset from {data_dir}")
    
    for class_idx, class_name in enumerate(class_folders):
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_dir} does not exist, skipping...")
            continue
        
        # Find all images
        images = []
        for ext in extensions:
            images.extend(list(class_dir.glob(f"*{ext}")))
            images.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        print(f"   {class_name}: {len(images)} images")
        
        for img_path in images:
            data.append((str(img_path), class_idx))
    
    print(f"✅ Total: {len(data)} images loaded")
    return data


def split_dataset(
    data: List[Tuple[str, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True
) -> Tuple[List, List, List]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        data: List of (image_path, label) tuples
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        stratify: Maintain class distribution in splits
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    if stratify:
        # Group by class
        class_data = {}
        for img_path, label in data:
            if label not in class_data:
                class_data[label] = []
            class_data[label].append((img_path, label))
        
        train_data, val_data, test_data = [], [], []
        
        for label, samples in class_data.items():
            random.shuffle(samples)
            n = len(samples)
            
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_data.extend(samples[:n_train])
            val_data.extend(samples[n_train:n_train + n_val])
            test_data.extend(samples[n_train + n_val:])
    else:
        # Random split
        data_copy = data.copy()
        random.shuffle(data_copy)
        
        n = len(data_copy)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_data = data_copy[:n_train]
        val_data = data_copy[n_train:n_train + n_val]
        test_data = data_copy[n_train + n_val:]
    
    # Shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print(f"\n📊 Dataset split:")
    print(f"   Training:   {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
    print(f"   Validation: {len(val_data)} samples ({len(val_data)/len(data)*100:.1f}%)")
    print(f"   Test:       {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
    
    return train_data, val_data, test_data


def analyze_dataset(data: List[Tuple[str, int]], class_names: List[str]):
    """Print dataset statistics"""
    labels = [label for _, label in data]
    counter = Counter(labels)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(data)}")
    print(f"   Number of classes: {len(class_names)}")
    print(f"\n   Class distribution:")
    
    for label, count in sorted(counter.items()):
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        percentage = count / len(data) * 100
        print(f"      {class_name}: {count} ({percentage:.1f}%)")


def save_dataset_split(
    train_data: List[Tuple[str, int]],
    val_data: List[Tuple[str, int]],
    test_data: List[Tuple[str, int]],
    save_path: Path,
    class_names: List[str]
):
    """Save dataset split to JSON file"""
    split_info = {
        'train': [{'path': path, 'label': label} for path, label in train_data],
        'val': [{'path': path, 'label': label} for path, label in val_data],
        'test': [{'path': path, 'label': label} for path, label in test_data],
        'class_names': class_names,
        'num_classes': len(class_names)
    }
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n💾 Dataset split saved to: {save_path}")


def load_dataset_split(load_path: Path) -> Tuple[List, List, List, List[str]]:
    """Load dataset split from JSON file"""
    with open(load_path, 'r') as f:
        split_info = json.load(f)
    
    train_data = [(item['path'], item['label']) for item in split_info['train']]
    val_data = [(item['path'], item['label']) for item in split_info['val']]
    test_data = [(item['path'], item['label']) for item in split_info['test']]
    class_names = split_info['class_names']
    
    print(f"✅ Dataset split loaded from: {load_path}")
    print(f"   Training: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    
    return train_data, val_data, test_data, class_names


def prepare_disease_dataset(
    data_dir: Path,
    save_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List, List, List]:
    """
    Prepare disease classification dataset
    
    Expected structure:
    data_dir/
        Normal/
        Pneumonia/
        Tuberculosis/
    
    Args:
        data_dir: Root directory with class folders
        save_dir: Directory to save split info
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    print(f"\n{'='*70}")
    print(f"Preparing Disease Classification Dataset")
    print(f"{'='*70}\n")
    
    # Load dataset
    data = load_dataset_from_folders(data_dir, DISEASE_CLASSES)
    
    # Analyze
    analyze_dataset(data, DISEASE_CLASSES)
    
    # Split
    train_data, val_data, test_data = split_dataset(
        data, train_ratio, val_ratio, test_ratio, stratify=True
    )
    
    # Save split
    save_path = Path(save_dir) / "disease_dataset_split.json"
    save_dataset_split(train_data, val_data, test_data, save_path, DISEASE_CLASSES)
    
    return train_data, val_data, test_data


def prepare_stage_dataset(
    data_dir: Path,
    save_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List, List, List]:
    """
    Prepare TB stage classification dataset
    
    Expected structure:
    data_dir/
        Stage_1/
        Stage_2/
        Stage_3/
        Stage_4/
        Stage_5/
    
    Args:
        data_dir: Root directory with stage folders
        save_dir: Directory to save split info
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    print(f"\n{'='*70}")
    print(f"Preparing TB Stage Classification Dataset")
    print(f"{'='*70}\n")
    
    # Load dataset
    stage_folders = [f"Stage_{i+1}" for i in range(len(TB_STAGE_CLASSES))]
    data = load_dataset_from_folders(data_dir, stage_folders)
    
    # Analyze
    analyze_dataset(data, TB_STAGE_CLASSES)
    
    # Split
    train_data, val_data, test_data = split_dataset(
        data, train_ratio, val_ratio, test_ratio, stratify=True
    )
    
    # Save split
    save_path = Path(save_dir) / "stage_dataset_split.json"
    save_dataset_split(train_data, val_data, test_data, save_path, TB_STAGE_CLASSES)
    
    return train_data, val_data, test_data


# Example usage
if __name__ == "__main__":
    print("🧪 Data Preparation Utilities")
    print("\nExample usage:")
    print("""
    # Prepare disease dataset
    train, val, test = prepare_disease_dataset(
        data_dir=Path("data/disease"),
        save_dir=Path("data/splits"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Prepare stage dataset
    train, val, test = prepare_stage_dataset(
        data_dir=Path("data/tb_stages"),
        save_dir=Path("data/splits"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Load saved split
    train, val, test, classes = load_dataset_split(
        Path("data/splits/disease_dataset_split.json")
    )
    """)
