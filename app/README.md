# Flask Web Application - Thermal Imaging Hotspot Detection

Flask-based web interface for the thermal imaging hotspot detection system.

## Features

- **Drag & Drop Upload**: Intuitive file upload with drag and drop support
- **Custom Save Location**: Choose where to save analysis results (Chrome/Edge browsers)
- **Real-time Processing**: Live feedback during image analysis
- **Multi-Model Pipeline**:
  - Equipment Classification (ResNet-50)
  - Hotspot Binary Classification (EfficientNet-B0)
  - Hotspot Detection (YOLOv11/YOLOv8)
- **Interactive Results**: Visual detection results with detailed analysis
- **Multiple Download Options**:
  - Standard download to browser's default location
  - Custom save location picker
- **Concurrent Request Handling**: Multiple users can process images simultaneously
- **Responsive Design**: Works on desktop and mobile devices

## Directory Structure

```
app/
├── app.py                  # Main Flask application
├── templates/
│   └── index.html         # Main web interface
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── main.js        # Frontend logic
├── uploads/               # Temporary uploaded images
└── results/               # Processed results
```

## Installation

1. **Install Flask dependencies:**
   ```bash
   pip install flask werkzeug
   ```

2. **Ensure models are trained:**
   - Equipment classification model in `models/equipment_classification/resnet50/`
   - Hotspot classification model in `models/hotspot_classification/efficientnet_b0/`
   - Detection model in `models/hotspot_detection/yolov11s/` or `yolov8s/`

## Usage

### Start the Application

```bash
# From project root
python app/app.py
```

Or use the provided script:
```bash
./bin/run_app.sh
```

The application will be available at: `http://localhost:5000`

### Using the Web Interface

1. **Select Save Location (Optional)**:
   - Click "Browse" button to select where results will be saved
   - Only works in Chrome/Edge browsers (File System Access API)
   - If not selected, downloads go to default browser location

2. **Upload Image**:
   - Drag and drop a thermal image onto the upload zone
   - Or click "Browse Files" to select an image
   - Supported formats: PNG, JPG, JPEG, BMP, TIF
   - Max file size: 16MB

3. **Analyze**:
   - Click "Analyze Image" button
   - Wait for processing (5-15 seconds)

4. **View Results**:
   - Detection visualization with bounding boxes
   - Equipment type classification
   - Hotspot presence classification
   - Detailed detection information

5. **Download Results**:
   - Click "Download Result" for default location
   - Click "Save to Custom Location" to choose save path
   
6. **New Analysis**:
   - Click "Analyze Another Image" to start over

## API Endpoints

### `GET /`
Main web interface

### `POST /upload`
Upload and process image

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
  "success": true,
  "filename": "20251130_120000_image.jpg",
  "result_image": "result_20251130_120000.jpg",
  "analysis": {
    "equipment": {
      "type": "Power_Transformers",
      "confidence": "95.23%"
    },
    "hotspot_classification": {
      "has_hotspot": true,
      "confidence": "92.15%"
    },
    "detections": {
      "count": 3,
      "hotspots": [
        {
          "bbox": [120, 80, 250, 180],
          "confidence": 0.89,
          "class": "hotspot"
        }
      ]
    }
  },
  "metadata": {
    "image_size": "1920x1080",
    "processed_at": "2025-11-30 12:00:00"
  }
}
```

### `GET /results/<filename>`
Get processed result image

### `GET /uploads/<filename>`
Get uploaded original image

### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "equipment": true,
    "hotspot": true,
    "detection": true
  }
}
```

## Configuration
## Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn

# Run with optimized settings for concurrent requests
gunicorn -w 4 --threads 2 --worker-class gthread -b 0.0.0.0:5000 --timeout 120 app.app:app
```

Or use the provided script:
```bash
./bin/run_app_production.sh
```

**Production Configuration:**
- **Workers**: 4 (for multi-core CPUs)
- **Threads per worker**: 2 (concurrent request handling)
- **Worker class**: gthread (supports threading)
- **Total concurrent connections**: 4 × 2 = 8 simultaneous users
- **Timeout**: 120 seconds (for ML model processing)
- **Binding**: 0.0.0.0:5000 (accessible from network)

This allows multiple users to upload and process images at the same time without blocking.
# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 app.app:app
```

Or use the provided script:
```bash
./bin/run_app_production.sh
```

## Troubleshooting

**Models not loading:**
- Ensure models are trained and checkpoints exist in `models/` directory
- Check console output for specific error messages

## Security Notes

**Current Security Features:**
- File upload limited to 16MB
- File type validation (MIME type checking)
- Secure filename handling
- Concurrent request handling with proper worker isolation
- Request timeout protection

**For Production Deployment, Add:**
- HTTPS/SSL certificates
- User authentication system
- Rate limiting to prevent abuse
- CORS policy configuration
- File upload virus scanning
- CSRF protection
- Reverse proxy (nginx/Apache)
- Firewall rules
- Environment variables for sensitive data
- First request may be slower due to model loading
- Subsequent requests are faster
- Consider using GPU for faster inference

## Security Notes

- File upload is limited to 16MB
- Only image file types are allowed
- Uploaded files are saved with secure filenames
- For production, add authentication and HTTPS

## License

Same as main project (see root LICENSE file)
