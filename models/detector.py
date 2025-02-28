"""
YOLOv8 model setup and training for Valorant UI detection
"""

import os
import torch
from ultralytics import YOLO

def setup_model(model_type="yolov8n.pt"):
    """Set up YOLOv8 model"""
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load pre-trained model
    model = YOLO(model_type)
    print(f"Loaded {model_type} model")
    
    return model

def train_model(model, yaml_path, epochs=50, img_size=640, batch_size=16, project_name="valorant_ui_detector"):
    """Train YOLOv8 model on Valorant UI dataset"""
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=project_name,
        patience=15,  # Early stopping patience
        device=0 if torch.cuda.is_available() else "cpu"
    )
    
    return results

def export_model(model, format='onnx'):
    """Export trained model to different formats"""
    # Valid formats: onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, torchscript
    model.export(format=format)
    print(f"Model exported to {format} format") 