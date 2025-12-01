#!/usr/bin/env python3
"""
Visualiza ejemplos de detección de hotspots con bounding boxes
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import json

def read_yolo_annotation(label_path):
    """Lee anotaciones en formato YOLO"""
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    boxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
    return boxes

def denormalize_box(box, img_width, img_height):
    """Convierte coordenadas normalizadas a píxeles"""
    x_center = box['x_center'] * img_width
    y_center = box['y_center'] * img_height
    width = box['width'] * img_width
    height = box['height'] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return x1, y1, x2, y2

def draw_boxes(image, boxes):
    """Dibuja bounding boxes en la imagen"""
    img_height, img_width = image.shape[:2]
    overlay = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = denormalize_box(box, img_width, img_height)
        
        # Dibujar rectángulo
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Añadir label
        label = 'Hotspot'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(overlay, 
                     (x1, y1 - label_size[1] - 4), 
                     (x1 + label_size[0], y1), 
                     (0, 255, 0), 
                     -1)
        cv2.putText(overlay, label, (x1, y1 - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return overlay

def main():
    # Configuración
    data_dir = Path('data/processed/hotspot_detection')
    raw_data_dir = Path('data/raw')  # Las imágenes están aquí
    labels_dir = data_dir / 'labels'
    stats_path = data_dir / 'annotation_stats.json'
    output_path = Path('visualizations/hotspot_detection_samples.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Leer estadísticas
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    print("=" * 60)
    print("Estadísticas de Detección de Hotspots")
    print("=" * 60)
    print(f"Total de imágenes procesadas: {stats['total_images']}")
    pct = (stats['images_with_hotspots'] / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
    print(f"Imágenes con hotspots: {stats['images_with_hotspots']} ({pct:.1f}%)")
    print(f"Total de hotspots detectados: {stats['total_hotspots']}")
    print(f"Promedio de hotspots por imagen: {stats['avg_hotspots_per_image']:.2f}")
    print("\nPor equipo:")
    for eq_name, eq_stats in stats['equipment_stats'].items():
        print(f"  {eq_name}:")
        print(f"    - Imágenes: {eq_stats['total']}")
        print(f"    - Con hotspots: {eq_stats['with_hotspots']}")
        print(f"    - Total hotspots: {eq_stats['total_hotspots']}")
    
    # Obtener imágenes con anotaciones
    annotated_images = []
    for label_file in labels_dir.glob('*.txt'):
        # Verificar que tenga anotaciones
        boxes = read_yolo_annotation(label_file)
        if len(boxes) > 0:
            # Buscar imagen en data/raw
            img_name = label_file.stem + '.jpg'
            img_path = None
            
            # Buscar en todos los subdirectorios de data/raw
            for equipment_dir in raw_data_dir.iterdir():
                if equipment_dir.is_dir():
                    potential_path = equipment_dir / img_name
                    if potential_path.exists():
                        img_path = potential_path
                        break
            
            if img_path and img_path.exists():
                annotated_images.append((img_path, label_file, len(boxes)))
    
    # Ordenar por número de hotspots y seleccionar ejemplos
    annotated_images.sort(key=lambda x: x[2], reverse=True)
    
    # Seleccionar 6 ejemplos: 2 con muchos hotspots, 2 medios, 2 con pocos
    n_samples = min(6, len(annotated_images))
    samples = []
    
    if n_samples >= 6:
        # 2 con más hotspots
        samples.extend(annotated_images[:2])
        # 2 del rango medio
        mid_start = len(annotated_images) // 3
        samples.extend(annotated_images[mid_start:mid_start+2])
        # 2 con menos hotspots
        samples.extend(annotated_images[-2:])
    else:
        samples = annotated_images[:n_samples]
    
    # Crear visualización
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (img_path, label_path, n_boxes) in enumerate(samples):
        # Leer imagen
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Leer anotaciones
        boxes = read_yolo_annotation(label_path)
        
        # Dibujar boxes
        image_with_boxes = draw_boxes(image, boxes)
        
        # Mostrar
        axes[idx].imshow(image_with_boxes)
        axes[idx].axis('off')
        
        # Título con información
        equipment = img_path.parent.name
        axes[idx].set_title(f'{equipment}\n{n_boxes} hotspot{"s" if n_boxes > 1 else ""} detectado{"s" if n_boxes > 1 else ""}',
                          fontsize=10, fontweight='bold')
    
    # Ocultar ejes sobrantes si hay menos de 6 imágenes
    for idx in range(len(samples), 6):
        axes[idx].axis('off')
    
    plt.suptitle('Ejemplos de Detección de Hotspots con Bounding Boxes', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualización guardada en: {output_path}")
    
    # Mostrar también algunos ejemplos de debug si existen
    debug_dir = data_dir / 'debug_detection'
    if debug_dir.exists():
        debug_images = list(debug_dir.glob('*.jpg'))
        if debug_images:
            print(f"\n✓ {len(debug_images)} imágenes de debug guardadas en: {debug_dir}")
            # Mostrar una imagen de debug de ejemplo
            sample_debug = random.choice(debug_images)
            debug_output = Path('visualizations/hotspot_detection_debug_sample.png')
            
            img = cv2.imread(str(sample_debug))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Debug: {sample_debug.stem}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(debug_output, dpi=150, bbox_inches='tight')
            print(f"✓ Ejemplo de debug guardado en: {debug_output}")

if __name__ == '__main__':
    main()
