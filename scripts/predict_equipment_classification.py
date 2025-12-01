#!/usr/bin/env python3
"""
Inference script for equipment classification.

This script loads a trained model and performs inference on new images.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.finetuning.models import get_model


def load_model(checkpoint_path: str, device: str = None) -> Tuple[torch.nn.Module, List[str], dict]:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, class_names, metrics)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get class names and number of classes
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    # Determine model type from checkpoint path
    if 'inception' in checkpoint_path.lower():
        model_type = 'inception_v3'
        input_size = 299
    else:
        model_type = 'resnet50'
        input_size = 224
    
    # Create model
    model = get_model(model_type, num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get metrics if available
    metrics = checkpoint.get('metrics', {})
    
    print(f"Loaded {model_type} model from {checkpoint_path}")
    print(f"Classes: {class_names}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    
    return model, class_names, metrics, input_size, device


def get_inference_transform(input_size: int) -> transforms.Compose:
    """
    Get transform for inference.
    
    Args:
        input_size: Input image size
    
    Returns:
        Transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str],
    transform: transforms.Compose,
    device: torch.device,
    top_k: int = 3
) -> dict:
    """
    Predict class for a single image.
    
    Args:
        model: Trained model
        image_path: Path to image file
        class_names: List of class names
        transform: Image transform
        device: Device to run inference on
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, min(top_k, len(class_names)))
    
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({
            'class': class_names[idx],
            'probability': float(prob),
            'confidence': f"{float(prob) * 100:.2f}%"
        })
    
    return {
        'image': image_path,
        'predictions': predictions,
        'top_class': predictions[0]['class'],
        'top_probability': predictions[0]['probability']
    }


def main():
    parser = argparse.ArgumentParser(
        description='Perform inference with trained equipment classification model'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image file'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Path to directory containing images'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top predictions to show (default: 3)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save predictions JSON file'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (auto-detected if not specified)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be specified")
    
    # Load model
    print("Loading model...")
    model, class_names, metrics, input_size, device = load_model(args.checkpoint, args.device)
    
    # Get transform
    transform = get_inference_transform(input_size)
    
    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(image_dir.glob(ext))
        image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        print("No images found!")
        return
    
    print(f"\nProcessing {len(image_paths)} image(s)...\n")
    
    # Run inference
    all_predictions = []
    for image_path in image_paths:
        try:
            result = predict_image(
                model, image_path, class_names, transform, device, args.top_k
            )
            all_predictions.append(result)
            
            # Print result
            print(f"Image: {Path(image_path).name}")
            print(f"Top prediction: {result['top_class']} ({result['predictions'][0]['confidence']})")
            
            if args.top_k > 1:
                print("All predictions:")
                for i, pred in enumerate(result['predictions'], 1):
                    print(f"  {i}. {pred['class']}: {pred['confidence']}")
            print()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        print(f"\nSaved predictions to {output_path}")
    
    # Summary
    if len(all_predictions) > 1:
        print("\nSummary:")
        class_counts = {}
        for pred in all_predictions:
            top_class = pred['top_class']
            class_counts[top_class] = class_counts.get(top_class, 0) + 1
        
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} image(s)")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")


if __name__ == '__main__':
    main()
