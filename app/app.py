"""
Flask Web Application for Thermal Imaging Hotspot Detection
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import json
import torch
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.hotspot_detector import HotspotDetectorPreprocessor
from src.finetuning.models import get_model as get_classification_model
from src.detection.models import YOLOv11Detector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# Create folders if they don't exist
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(exist_ok=True)

# Equipment class names
EQUIPMENT_CLASSES = [
    'Circuit_Breakers',
    'Disconnectors', 
    'Power_Transformers',
    'Surge_Arresters',
    'Wave_Traps'
]

# Global model instances (loaded on first use)
equipment_model = None
hotspot_model = None
detection_model = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_models():
    """Load all models (lazy loading)."""
    global equipment_model, hotspot_model, detection_model
    
    if equipment_model is None:
        try:
            # Load equipment classification model (ResNet-50)
            checkpoint_path = project_root / 'models/equipment_classification/resnet50/best_model.pth'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                equipment_model = get_classification_model('resnet50', len(EQUIPMENT_CLASSES), pretrained=False)
                equipment_model.load_state_dict(checkpoint['model_state_dict'])
                equipment_model.eval()
                print("✅ Equipment classification model loaded")
        except Exception as e:
            print(f"⚠️ Could not load equipment model: {e}")
    
    if hotspot_model is None:
        try:
            # Load hotspot classification model (EfficientNet-B0)
            from src.finetuning.hotspot_models import get_hotspot_model
            checkpoint_path = project_root / 'models/hotspot_classification/efficientnet_b0/best_model.pth'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                hotspot_model = get_hotspot_model('efficientnet_b0', 2, pretrained=False)
                hotspot_model.load_state_dict(checkpoint['model_state_dict'])
                hotspot_model.eval()
                print("✅ Hotspot classification model loaded")
        except Exception as e:
            print(f"⚠️ Could not load hotspot model: {e}")
    
    if detection_model is None:
        try:
            # Load detection model (YOLOv11)
            weights_path = project_root / 'models/hotspot_detection/yolov11s/weights/best.pt'
            if weights_path.exists():
                from ultralytics import YOLO
                detection_model = YOLO(str(weights_path))
                print("✅ Detection model loaded")
            else:
                # Fallback to YOLOv8
                weights_path = project_root / 'models/hotspot_detection/yolov8s/weights/best.pt'
                if weights_path.exists():
                    from ultralytics import YOLO
                    detection_model = YOLO(str(weights_path))
                    print("✅ Detection model (YOLOv8) loaded")
        except Exception as e:
            print(f"⚠️ Could not load detection model: {e}")


def preprocess_image(image_path, target_size=(416, 416)):
    """Preprocess image for detection."""
    preprocessor = HotspotDetectorPreprocessor(
        input_dir=str(image_path.parent),
        output_dir=str(app.config['RESULTS_FOLDER'] / 'preprocessed'),
        target_size=target_size
    )
    
    # Load and preprocess
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read image")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def classify_equipment(image):
    """Classify equipment type."""
    if equipment_model is None:
        return None, None
    
    # Preprocess for ResNet-50
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = equipment_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    return EQUIPMENT_CLASSES[predicted_class], confidence


def classify_hotspot(image):
    """Classify if image has hotspot."""
    if hotspot_model is None:
        return None, None
    
    # Preprocess for EfficientNet-B0
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = hotspot_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        has_hotspot = probabilities[1].item() > probabilities[0].item()
        confidence = probabilities[1 if has_hotspot else 0].item()
    
    return has_hotspot, confidence


def detect_hotspots(image_path):
    """Detect hotspots using YOLO."""
    if detection_model is None:
        return None, None
    
    # Run detection
    results = detection_model.predict(
        source=str(image_path),
        conf=0.25,
        iou=0.45,
        save=False,
        verbose=False
    )
    
    # Get detections
    detections = []
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            
            detections.append({
                'bbox': [int(x) for x in box],
                'confidence': float(conf),
                'class': 'hotspot'
            })
    
    # Draw detections on image
    img = cv2.imread(str(image_path))
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Hotspot {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Save result image
    result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    result_path = app.config['RESULTS_FOLDER'] / result_filename
    cv2.imwrite(str(result_path), img)
    
    return detections, result_filename


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, TIF'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(filepath))
        
        # Load models
        load_models()
        
        # Read image
        img = cv2.imread(str(filepath))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Equipment Classification
        equipment_type, equipment_conf = classify_equipment(img_rgb)
        
        # 2. Hotspot Classification
        has_hotspot, hotspot_conf = classify_hotspot(img_rgb)
        
        # 3. Hotspot Detection
        detections, result_filename = detect_hotspots(filepath)
        
        # Prepare response
        response = {
            'success': True,
            'filename': filename,
            'result_image': result_filename,
            'analysis': {
                'equipment': {
                    'type': equipment_type,
                    'confidence': f"{equipment_conf * 100:.2f}%" if equipment_conf else 'N/A'
                },
                'hotspot_classification': {
                    'has_hotspot': has_hotspot,
                    'confidence': f"{hotspot_conf * 100:.2f}%" if hotspot_conf else 'N/A'
                },
                'detections': {
                    'count': len(detections) if detections else 0,
                    'hotspots': detections if detections else []
                }
            },
            'metadata': {
                'image_size': f"{img.shape[1]}x{img.shape[0]}",
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results/<filename>')
def get_result(filename):
    """Serve result images."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/uploads/<filename>')
def get_upload(filename):
    """Serve uploaded images."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models': {
            'equipment': equipment_model is not None,
            'hotspot': hotspot_model is not None,
            'detection': detection_model is not None
        }
    })


if __name__ == '__main__':
    print("🚀 Starting Thermal Imaging Hotspot Detection App")
    print("📍 Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
