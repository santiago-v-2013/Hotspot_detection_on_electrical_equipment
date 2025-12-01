// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const browseSaveBtn = document.getElementById('browseSaveBtn');
const saveLocationInput = document.getElementById('saveLocation');
const clearBtn = document.getElementById('clearBtn');
const processBtn = document.getElementById('processBtn');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const loading = document.getElementById('loading');
const loadingSteps = document.getElementById('loadingSteps');
const resultsSection = document.getElementById('resultsSection');
const errorToast = document.getElementById('errorToast');
const errorMessage = document.getElementById('errorMessage');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

// State
let selectedFile = null;
let currentResultPath = null;
let customSavePath = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

function setupEventListeners() {
    // Browse button
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Browse save location button
    browseSaveBtn.addEventListener('click', async () => {
        try {
            // Use File System Access API if available (Chrome, Edge)
            if ('showDirectoryPicker' in window) {
                const dirHandle = await window.showDirectoryPicker({
                    mode: 'readwrite'
                });
                customSavePath = dirHandle;
                saveLocationInput.value = dirHandle.name;
                showSuccess('Save location selected: ' + dirHandle.name);
            } else {
                // Fallback for browsers without File System Access API
                showError('Custom save location requires a modern browser (Chrome/Edge)');
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('Error selecting save location:', error);
                showError('Failed to select save location');
            }
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        handleFileSelect(file);
    });

    // Clear button
    clearBtn.addEventListener('click', clearImage);

    // Process button
    processBtn.addEventListener('click', processImage);

    // New analysis button
    newAnalysisBtn.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        document.querySelector('.upload-section').scrollIntoView({ behavior: 'smooth' });
        clearImage();
    });
}

function handleFileSelect(file) {
    if (!file) return;

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload PNG, JPG, JPEG, BMP, or TIF image.');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size exceeds 16MB limit.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        document.querySelector('.drop-zone-content').style.display = 'none';
        imagePreview.style.display = 'block';
        processBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    document.querySelector('.drop-zone-content').style.display = 'block';
    imagePreview.style.display = 'none';
    processBtn.disabled = true;
}

async function processImage() {
    if (!selectedFile) return;

    // Show loading
    loading.style.display = 'block';
    processBtn.disabled = true;
    resultsSection.style.display = 'none';

    // Simulate processing steps
    const steps = [
        'Uploading image...',
        'Loading AI models...',
        'Classifying equipment type...',
        'Analyzing hotspot presence...',
        'Detecting hotspot locations...',
        'Generating results...'
    ];

    let stepIndex = 0;
    const stepInterval = setInterval(() => {
        if (stepIndex < steps.length) {
            loadingSteps.textContent = steps[stepIndex];
            stepIndex++;
        }
    }, 800);

    try {
        // Upload and process
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        clearInterval(stepInterval);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Processing failed');
        }

        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            throw new Error('Processing failed');
        }

    } catch (error) {
        clearInterval(stepInterval);
        showError(error.message);
        processBtn.disabled = false;
    } finally {
        loading.style.display = 'none';
    }
}

function displayResults(data) {
    const { analysis, result_image, metadata } = data;

    // Result image
    const resultImage = document.getElementById('resultImage');
    resultImage.src = `/results/${result_image}`;
    currentResultPath = `/results/${result_image}`;
    
    // Setup download buttons
    setupDownloadButtons();

    // Equipment classification
    const equipmentType = document.getElementById('equipmentType');
    const equipmentConf = document.getElementById('equipmentConf');
    const equipmentConfBar = document.getElementById('equipmentConfBar');
    
    if (analysis.equipment.type) {
        equipmentType.textContent = analysis.equipment.type.replace(/_/g, ' ');
        equipmentConf.textContent = analysis.equipment.confidence;
        const confValue = parseFloat(analysis.equipment.confidence);
        equipmentConfBar.style.width = confValue + '%';
    } else {
        equipmentType.textContent = 'N/A';
        equipmentConf.textContent = '-';
        equipmentConfBar.style.width = '0%';
    }

    // Hotspot classification
    const hotspotStatus = document.getElementById('hotspotStatus');
    const hotspotConf = document.getElementById('hotspotConf');
    const hotspotConfBar = document.getElementById('hotspotConfBar');
    
    if (analysis.hotspot_classification.has_hotspot !== null) {
        const hasHotspot = analysis.hotspot_classification.has_hotspot;
        hotspotStatus.textContent = hasHotspot ? 'Hotspot Detected' : 'No Hotspot';
        hotspotStatus.className = 'value status ' + (hasHotspot ? 'positive' : 'negative');
        hotspotConf.textContent = analysis.hotspot_classification.confidence;
        const confValue = parseFloat(analysis.hotspot_classification.confidence);
        hotspotConfBar.style.width = confValue + '%';
    } else {
        hotspotStatus.textContent = 'N/A';
        hotspotConf.textContent = '-';
        hotspotConfBar.style.width = '0%';
    }

    // Detection count
    const detectionCount = document.getElementById('detectionCount');
    detectionCount.textContent = analysis.detections.count;

    // Detections list
    const detectionsList = document.getElementById('detectionsList');
    detectionsList.innerHTML = '';
    
    if (analysis.detections.hotspots && analysis.detections.hotspots.length > 0) {
        analysis.detections.hotspots.forEach((det, index) => {
            const detItem = document.createElement('div');
            detItem.className = 'detection-item';
            detItem.innerHTML = `
                <div class="detection-info">
                    <span><strong>Hotspot ${index + 1}</strong></span>
                    <span>${(det.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="detection-bbox">
                    Location: [${det.bbox[0]}, ${det.bbox[1]}, ${det.bbox[2]}, ${det.bbox[3]}]
                </div>
            `;
            detectionsList.appendChild(detItem);
        });
    } else {
        detectionsList.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.875rem;">No hotspots detected</p>';
    }

    // Metadata
    document.getElementById('imageSize').textContent = metadata.image_size;
    document.getElementById('processedAt').textContent = metadata.processed_at;

    // Show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    errorMessage.textContent = message;
    errorToast.style.display = 'flex';
    
    setTimeout(() => {
        errorToast.style.display = 'none';
    }, 3000);
}

// Show success message
function showSuccess(message) {
    const toast = document.createElement('div');
    toast.className = 'toast toast-success';
    toast.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span>${message}</span>
    `;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('show');
    }, 100);

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function setupDownloadButtons() {
    // Standard download button
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.onclick = () => {
        if (currentResultPath) {
            const link = document.createElement('a');
            link.href = currentResultPath;
            link.download = `hotspot_detection_${Date.now()}.jpg`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    // Custom location download button
    const downloadCustomBtn = document.getElementById('downloadCustomBtn');
    downloadCustomBtn.onclick = async () => {
        if (!currentResultPath) return;

        try {
            if (customSavePath) {
                // Save to previously selected directory
                const response = await fetch(currentResultPath);
                const blob = await response.blob();
                const filename = `hotspot_detection_${Date.now()}.jpg`;
                const fileHandle = await customSavePath.getFileHandle(filename, { create: true });
                const writable = await fileHandle.createWritable();
                await writable.write(blob);
                await writable.close();
                showSuccess(`Saved to ${customSavePath.name}/${filename}`);
            } else if ('showSaveFilePicker' in window) {
                // Show save dialog
                const response = await fetch(currentResultPath);
                const blob = await response.blob();
                const fileHandle = await window.showSaveFilePicker({
                    suggestedName: `hotspot_detection_${Date.now()}.jpg`,
                    types: [{
                        description: 'JPEG Image',
                        accept: { 'image/jpeg': ['.jpg', '.jpeg'] }
                    }]
                });
                const writable = await fileHandle.createWritable();
                await writable.write(blob);
                await writable.close();
                showSuccess('File saved successfully');
            } else {
                // Fallback to standard download
                showError('Custom save location requires a modern browser (Chrome/Edge)');
                downloadBtn.click();
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('Error saving file:', error);
                showError('Failed to save file to custom location');
            }
        }
    };
}
