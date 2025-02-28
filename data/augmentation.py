"""
Data augmentation utilities for Valorant UI detection
"""

import os
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
import cv2

def basic_augmentation(dataset_dir, augmentations_per_image=3):
    """Apply basic augmentations to training images"""
    train_imgs = glob(os.path.join(dataset_dir, "images/train/*.png"))
    
    print(f"Applying {augmentations_per_image} augmentations to {len(train_imgs)} images...")
    
    for img_path in tqdm(train_imgs):
        img = cv2.imread(img_path)
        base_name = os.path.basename(img_path).split('.')[0]
        label_path = img_path.replace("images", "labels").replace(".png", ".txt")
        
        if not os.path.exists(label_path):
            print(f"Warning: No label file for {img_path}, skipping augmentation")
            continue
        
        # Apply different augmentations
        for i in range(augmentations_per_image):
            # Choose random augmentation
            aug_type = np.random.choice(['brightness', 'contrast', 'blur', 'noise'])
            
            if aug_type == 'brightness':
                factor = np.random.uniform(0.5, 1.5)
                img_aug = cv2.convertScaleAbs(img, alpha=factor, beta=0)
            elif aug_type == 'contrast':
                factor = np.random.uniform(0.5, 2.0)
                img_aug = cv2.convertScaleAbs(img, alpha=factor, beta=128*(1-factor))
            elif aug_type == 'blur':
                kernel_size = np.random.choice([3, 5, 7])
                img_aug = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            elif aug_type == 'noise':
                noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
                img_aug = cv2.add(img, noise)
            
            # Save augmented image
            aug_img_path = os.path.join(dataset_dir, f"images/train/{base_name}_aug{i}.png")
            cv2.imwrite(aug_img_path, img_aug)
            
            # Copy the label file (annotations remain the same)
            aug_label_path = os.path.join(dataset_dir, f"labels/train/{base_name}_aug{i}.txt")
            shutil.copy(label_path, aug_label_path)
    
    # Count total images after augmentation
    train_imgs_after = glob(os.path.join(dataset_dir, "images/train/*.png"))
    print(f"Dataset size after augmentation: {len(train_imgs_after)} training images")
    
    return len(train_imgs_after) 