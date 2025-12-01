#!/usr/bin/env python3
"""
Prediction script for hotspot classification.

This script loads a trained model and performs inference on new images
to classify them as hotspot or no_hotspot.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torchvision import transforms
from PIL import Image
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.finetuning.hotspot_models import get_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, model_name: str, device: str = 'cuda') -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pth file)
        model_name: Name of the model architecture
        device: Device to load the model on
        
    Returns:
        Loaded model in eval mode
    """
    # Create model
    model = get_model(
        model_name=model_name,
        num_classes=2,
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    logger.info(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown'):.4f}")
    
    return model


def get_transform(image_size: int = 224):
    """Get image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    transform: transforms.Compose,
    device: str = 'cuda',
    class_names: List[str] = ['no_hotspot', 'hotspot']
) -> Tuple[str, float, dict]:
    """
    Predict the class of a single image.
    
    Args:
        model: Trained model
        image_path: Path to the image
        transform: Image transformation pipeline
        device: Device to run inference on
        class_names: List of class names
        
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
    
    predicted_class = class_names[predicted_idx]
    all_probs = {
        class_name: prob.item()
        for class_name, prob in zip(class_names, probabilities)
    }
    
    return predicted_class, confidence, all_probs


def predict_batch(
    model: torch.nn.Module,
    image_paths: List[str],
    transform: transforms.Compose,
    device: str = 'cuda',
    class_names: List[str] = ['no_hotspot', 'hotspot']
) -> List[dict]:
    """
    Predict classes for multiple images.
    
    Args:
        model: Trained model
        image_paths: List of image paths
        transform: Image transformation pipeline
        device: Device to run inference on
        class_names: List of class names
        
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    for image_path in image_paths:
        try:
            predicted_class, confidence, all_probs = predict_image(
                model, image_path, transform, device, class_names
            )
            
            results.append({
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': all_probs
            })
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description='Predict hotspot classification for images'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['efficientnet_b0', 'densenet121'],
        required=True,
        help='Model architecture'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image to classify'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Path to directory containing images to classify'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be specified")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.model, args.device)
    
    # Get transform
    transform = get_transform(image_size=224)
    
    # Get image paths
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = (
            list(image_dir.glob('*.jpg')) +
            list(image_dir.glob('*.png')) +
            list(image_dir.glob('*.jpeg'))
        )
        image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        logger.error("No images found")
        return
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Predict
    class_names = ['no_hotspot', 'hotspot']
    results = predict_batch(model, image_paths, transform, args.device, class_names)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 80)
    
    for result in results:
        if 'error' in result:
            logger.error(f"{result['image_path']}: ERROR - {result['error']}")
        else:
            logger.info(f"\nImage: {result['image_path']}")
            logger.info(f"  Prediction: {result['predicted_class']}")
            logger.info(f"  Confidence: {result['confidence']:.4f}")
            logger.info(f"  Probabilities:")
            for class_name, prob in result['probabilities'].items():
                logger.info(f"    {class_name}: {prob:.4f}")
    
    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {output_path}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    successful = [r for r in results if 'error' not in r]
    if successful:
        hotspot_count = sum(1 for r in successful if r['predicted_class'] == 'hotspot')
        no_hotspot_count = len(successful) - hotspot_count
        
        logger.info(f"Total images: {len(successful)}")
        logger.info(f"Hotspot: {hotspot_count} ({hotspot_count/len(successful)*100:.1f}%)")
        logger.info(f"No hotspot: {no_hotspot_count} ({no_hotspot_count/len(successful)*100:.1f}%)")
    
    if len(successful) < len(results):
        logger.warning(f"Failed: {len(results) - len(successful)} images")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")


if __name__ == '__main__':
    main()
