"""
Dataset preparation utilities for Valorant UI detection
"""

import os
import cv2
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm
import yaml

def setup_dataset_directories(dataset_dir):
    """Create directories for our dataset"""
    os.makedirs(os.path.join(dataset_dir, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels/val"), exist_ok=True)
    
    return dataset_dir

def collect_screenshots(source_dir, dataset_dir, split_ratio=0.8):
    """
    Collect screenshots from source_dir and split into train/val sets
    """
    screenshots = glob(os.path.join(source_dir, "*.png"))
    if not screenshots:
        screenshots = glob(os.path.join(source_dir, "*.jpg"))
    
    if not screenshots:
        raise ValueError(f"No images found in {source_dir}")
    
    np.random.shuffle(screenshots)
    
    split_idx = int(len(screenshots) * split_ratio)
    train_imgs = screenshots[:split_idx]
    val_imgs = screenshots[split_idx:]
    
    # Copy images to dataset directories
    print("Copying training images...")
    for i, img_path in enumerate(tqdm(train_imgs)):
        img = cv2.imread(img_path)
        cv2.imwrite(f"{dataset_dir}/images/train/img_{i:04d}.png", img)
    
    print("Copying validation images...")
    for i, img_path in enumerate(tqdm(val_imgs)):
        img = cv2.imread(img_path)
        cv2.imwrite(f"{dataset_dir}/images/val/img_{i:04d}.png", img)
    
    return len(train_imgs), len(val_imgs)

def create_dataset_yaml(dataset_dir, classes):
    """Create YAML configuration file for YOLOv8 training"""
    yaml_content = f"""
# Valorant UI Detection Dataset
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val

# Classes
names:
"""
    
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"
    
    yaml_path = os.path.join(dataset_dir, "valorant_ui.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    return yaml_path

def check_annotations(dataset_dir):
    """Check if annotations exist for all images"""
    train_images = glob(os.path.join(dataset_dir, "images/train/*.png"))
    val_images = glob(os.path.join(dataset_dir, "images/val/*.png"))
    
    missing_annotations = []
    
    for img_path in train_images + val_images:
        label_path = img_path.replace("images", "labels").replace(".png", ".txt")
        if not os.path.exists(label_path):
            missing_annotations.append(img_path)
    
    if missing_annotations:
        print(f"Warning: {len(missing_annotations)} images are missing annotations.")
        print("First 5 missing annotations:")
        for path in missing_annotations[:5]:
            print(f"  - {path}")
        
        return False
    
    return True 