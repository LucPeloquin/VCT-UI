"""
Visualization utilities for Valorant UI detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def draw_detections(image, detections, class_names, conf_threshold=0.25):
    """Draw bounding boxes and labels on the image"""
    img_copy = image.copy()
    
    # Generate random colors for each class
    np.random.seed(42)  # For reproducibility
    colors = {i: tuple(map(int, np.random.randint(0, 255, 3))) for i in range(len(class_names))}
    
    for det in detections:
        if det.conf[0] < conf_threshold:
            continue
            
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        cls_id = int(det.cls[0])
        conf = float(det.conf[0])
        
        # Get class name and color
        cls_name = class_names[cls_id]
        color = colors[cls_id]
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        text = f"{cls_name} {conf:.2f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_copy, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(img_copy, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_copy

def plot_training_results(results_file):
    """Plot training metrics from results.csv"""
    import pandas as pd
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    # Load results
    results = pd.read_csv(results_file)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot training and validation loss
    axs[0, 0].plot(results['epoch'], results['train/box_loss'], label='train')
    if 'val/box_loss' in results.columns:
        axs[0, 0].plot(results['epoch'], results['val/box_loss'], label='val')
    axs[0, 0].set_title('Box Loss')
    axs[0, 0].legend()
    
    # Plot mAP50
    if 'metrics/mAP50(B)' in results.columns:
        axs[0, 1].plot(results['epoch'], results['metrics/mAP50(B)'])
        axs[0, 1].set_title('mAP50')
    
    # Plot mAP50-95
    if 'metrics/mAP50-95(B)' in results.columns:
        axs[1, 0].plot(results['epoch'], results['metrics/mAP50-95(B)'])
        axs[1, 0].set_title('mAP50-95')
    
    # Plot learning rate
    if 'lr/pg0' in results.columns:
        axs[1, 1].plot(results['epoch'], results['lr/pg0'])
        axs[1, 1].set_title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()
    
    print(f"Training results plot saved to training_results.png") 