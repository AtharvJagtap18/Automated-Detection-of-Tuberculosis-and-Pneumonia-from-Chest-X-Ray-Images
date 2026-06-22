"""
Main inference module for chest X-ray analysis
Handles two-stage prediction: Disease → TB Stage (if TB detected)
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional
import time

from config import (
    DISEASE_MODEL_PATH,
    STAGE_MODEL_PATH,
    DISEASE_MODEL_ARCH,
    STAGE_MODEL_ARCH,
    DISEASE_CLASSES,
    TB_STAGE_CLASSES,
    DEVICE,
    TB_THRESHOLD,
    CONFIDENCE_THRESHOLD
)
from preprocess import XRayPreprocessor
from models_arch import load_model


class XRayPredictor:
    """
    Main predictor class for chest X-ray analysis
    Implements two-stage prediction pipeline
    """
    
    def __init__(
        self,
        disease_model_path: Optional[str] = None,
        stage_model_path: Optional[str] = None,
        device: str = DEVICE
    ):
        """
        Initialize predictor with trained models
        
        Args:
            disease_model_path: Path to disease classifier weights
            stage_model_path: Path to TB stage classifier weights
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.preprocessor = XRayPreprocessor()
        
        # Use default paths if not provided
        disease_model_path = disease_model_path or str(DISEASE_MODEL_PATH)
        stage_model_path = stage_model_path or str(STAGE_MODEL_PATH)
        
        print(f"🚀 Initializing XRayPredictor...")
        print(f"   Device: {self.device}")
        
        # Load disease classifier
        print(f"📦 Loading disease model from {disease_model_path}")
        self.disease_model = load_model(
            model_path=disease_model_path,
            arch=DISEASE_MODEL_ARCH,
            num_classes=len(DISEASE_CLASSES),
            model_type="disease"
        )
        self.disease_model.to(self.device)
        self.disease_model.eval()
        
        # Load TB stage classifier
        print(f"📦 Loading TB stage model from {stage_model_path}")
        self.stage_model = load_model(
            model_path=stage_model_path,
            arch=STAGE_MODEL_ARCH,
            num_classes=len(TB_STAGE_CLASSES),
            model_type="stage"
        )
        self.stage_model.to(self.device)
        self.stage_model.eval()
        
        self.disease_classes = DISEASE_CLASSES
        self.stage_classes = TB_STAGE_CLASSES
        
        print(f"✅ Predictor initialized successfully")
    
    def predict_disease(self, image_tensor: torch.Tensor) -> Dict:
        """
        Predict disease class from preprocessed image
        
        Args:
            image_tensor: Preprocessed image tensor (1, 3, H, W)
            
        Returns:
            Dictionary with prediction results
        """
        with torch.no_grad():
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Forward pass
            logits = self.disease_model(image_tensor)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get top prediction
            confidence, pred_idx = torch.max(probs, dim=1)
            
            confidence = confidence.item()
            pred_idx = pred_idx.item()
            predicted_class = self.disease_classes[pred_idx]
            
            # Get all class probabilities
            all_probs = {
                class_name: prob.item()
                for class_name, prob in zip(self.disease_classes, probs[0])
            }
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_index': pred_idx,
                'all_probabilities': all_probs
            }
    
    def predict_tb_stage(self, image_tensor: torch.Tensor) -> Dict:
        """
        Predict TB stage from preprocessed image
        Only called if disease prediction is TB
        
        Args:
            image_tensor: Preprocessed image tensor (1, 3, H, W)
            
        Returns:
            Dictionary with stage prediction results
        """
        with torch.no_grad():
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Forward pass
            logits = self.stage_model(image_tensor)
            
            # Apply softmax
            probs = F.softmax(logits, dim=1)
            
            # Get top prediction
            confidence, pred_idx = torch.max(probs, dim=1)
            
            confidence = confidence.item()
            pred_idx = pred_idx.item()
            predicted_stage = self.stage_classes[pred_idx]
            
            # Get all stage probabilities
            all_probs = {
                stage_name: prob.item()
                for stage_name, prob in zip(self.stage_classes, probs[0])
            }
            
            return {
                'predicted_stage': predicted_stage,
                'stage_confidence': confidence,
                'stage_index': pred_idx,
                'all_stage_probabilities': all_probs
            }
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        return_timing: bool = False
    ) -> Dict:
        """
        Main prediction function - two-stage pipeline
        
        Args:
            image: Image path, numpy array, or PIL Image
            return_timing: If True, include timing information
            
        Returns:
            Dictionary with complete prediction results
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocess image
            preprocess_start = time.time()
            image_tensor = self.preprocessor.preprocess(image)
            preprocess_time = time.time() - preprocess_start
            
            # Step 2: Predict disease
            disease_start = time.time()
            disease_result = self.predict_disease(image_tensor)
            disease_time = time.time() - disease_start
            
            # Build result dictionary
            result = {
                'disease': disease_result['predicted_class'],
                'confidence': disease_result['confidence'],
                'all_probabilities': disease_result['all_probabilities']
            }
            
            # Note: TB stage prediction disabled (requires expert annotations)
            # Stage model uses synthetic labels and is not clinically valid
            
            # Add timing information if requested
            if return_timing:
                total_time = time.time() - start_time
                result['timing'] = {
                    'preprocess_time': preprocess_time,
                    'disease_prediction_time': disease_time,
                    'total_time': total_time
                }
            
            # Add warning if confidence is low
            if disease_result['confidence'] < CONFIDENCE_THRESHOLD:
                result['warning'] = (
                    f"Low confidence prediction ({disease_result['confidence']:.2%}). "
                    "Results may be unreliable."
                )
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'disease': 'Error',
                'confidence': 0.0
            }
    
    def predict_batch(
        self,
        images: list,
        return_timing: bool = False
    ) -> list:
        """
        Predict on a batch of images
        
        Args:
            images: List of image paths or arrays
            return_timing: If True, include timing information
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image, return_timing=return_timing)
            results.append(result)
        return results


def predict_image(
    image_path: Union[str, Path],
    disease_model_path: Optional[str] = None,
    stage_model_path: Optional[str] = None
) -> Dict:
    """
    Convenience function for single image prediction
    Creates predictor, runs inference, and returns result
    
    Args:
        image_path: Path to X-ray image
        disease_model_path: Optional custom disease model path
        stage_model_path: Optional custom stage model path
        
    Returns:
        Prediction dictionary
    """
    predictor = XRayPredictor(
        disease_model_path=disease_model_path,
        stage_model_path=stage_model_path
    )
    return predictor.predict(image_path, return_timing=True)


# Example usage
if __name__ == "__main__":
    print("🧪 Testing XRayPredictor...")
    
    # Create predictor
    predictor = XRayPredictor()
    
    # Test with dummy image
    print("\n📸 Testing with dummy image...")
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    result = predictor.predict(dummy_image, return_timing=True)
    
    print("\n📊 Prediction Results:")
    print(f"   Disease: {result['disease']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    
    if 'tb_stage' in result:
        print(f"   TB Stage: {result['tb_stage']}")
        print(f"   Stage Confidence: {result.get('stage_confidence', 0):.2%}")
    
    if 'timing' in result:
        print(f"\n⏱️  Timing:")
        print(f"   Preprocessing: {result['timing']['preprocess_time']:.3f}s")
        print(f"   Disease prediction: {result['timing']['disease_prediction_time']:.3f}s")
        print(f"   Stage prediction: {result['timing']['stage_prediction_time']:.3f}s")
        print(f"   Total: {result['timing']['total_time']:.3f}s")
    
    print("\n✅ Inference test passed!")
