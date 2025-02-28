"""
Video processing utilities for Valorant UI detection
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from a video at specified intervals
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
    
    Returns:
        Number of frames extracted
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Extracting every {frame_interval} frames...")
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames//frame_interval) as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Save frame
                frame_path = os.path.join(output_dir, f"frame_{saved_count:05d}.png")
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                pbar.update(1)
            
            frame_count += 1
    
    # Release video
    cap.release()
    
    print(f"Extracted {saved_count} frames to {output_dir}")
    return saved_count

def apply_static_annotations(template_annotation_path, frames_dir, labels_dir):
    """
    Apply the same annotation to all frames
    
    Args:
        template_annotation_path: Path to the template annotation file (YOLO format)
        frames_dir: Directory containing extracted frames
        labels_dir: Directory to save annotation files
    
    Returns:
        Number of annotation files created
    """
    # Create labels directory if it doesn't exist
    os.makedirs(labels_dir, exist_ok=True)
    
    # Check if template annotation exists
    if not os.path.exists(template_annotation_path):
        print(f"Error: Template annotation file not found at {template_annotation_path}")
        return 0
    
    # Get all frames
    frames = [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))]
    
    # Read template annotation
    with open(template_annotation_path, 'r') as f:
        template_content = f.read()
    
    # Apply to all frames
    count = 0
    for frame in tqdm(frames, desc="Applying annotations"):
        # Create annotation filename (same basename as frame but .txt extension)
        base_name = os.path.splitext(frame)[0]
        annotation_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        # Copy template content
        with open(annotation_path, 'w') as f:
            f.write(template_content)
        
        count += 1
    
    print(f"Applied annotations to {count} frames")
    return count

def prepare_video_dataset(video_path, dataset_dir, template_annotation_path, frame_interval=30, train_ratio=0.8):
    """
    Prepare a dataset from video with static annotations
    
    Args:
        video_path: Path to the video file
        dataset_dir: Base directory for the dataset
        template_annotation_path: Path to the template annotation file
        frame_interval: Extract every Nth frame
        train_ratio: Ratio of frames to use for training (vs validation)
    
    Returns:
        Dictionary with dataset statistics
    """
    # Create temporary directory for extracted frames
    temp_frames_dir = os.path.join(dataset_dir, "temp_frames")
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    # Create dataset directories
    train_images_dir = os.path.join(dataset_dir, "images/train")
    val_images_dir = os.path.join(dataset_dir, "images/val")
    train_labels_dir = os.path.join(dataset_dir, "labels/train")
    val_labels_dir = os.path.join(dataset_dir, "labels/val")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Extract frames
    num_frames = extract_frames(video_path, temp_frames_dir, frame_interval)
    
    if num_frames == 0:
        print("No frames extracted. Exiting.")
        return {"train": 0, "val": 0}
    
    # Apply annotations to all frames
    temp_labels_dir = os.path.join(dataset_dir, "temp_labels")
    os.makedirs(temp_labels_dir, exist_ok=True)
    apply_static_annotations(template_annotation_path, temp_frames_dir, temp_labels_dir)
    
    # Split into train/val
    frames = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith(('.png', '.jpg'))])
    np.random.shuffle(frames)
    
    split_idx = int(len(frames) * train_ratio)
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]
    
    # Copy to train/val directories
    print("Copying files to train directory...")
    for frame in tqdm(train_frames):
        # Copy image
        src_img = os.path.join(temp_frames_dir, frame)
        dst_img = os.path.join(train_images_dir, frame)
        shutil.copy(src_img, dst_img)
        
        # Copy annotation
        base_name = os.path.splitext(frame)[0]
        src_label = os.path.join(temp_labels_dir, f"{base_name}.txt")
        dst_label = os.path.join(train_labels_dir, f"{base_name}.txt")
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
    
    print("Copying files to validation directory...")
    for frame in tqdm(val_frames):
        # Copy image
        src_img = os.path.join(temp_frames_dir, frame)
        dst_img = os.path.join(val_images_dir, frame)
        shutil.copy(src_img, dst_img)
        
        # Copy annotation
        base_name = os.path.splitext(frame)[0]
        src_label = os.path.join(temp_labels_dir, f"{base_name}.txt")
        dst_label = os.path.join(val_labels_dir, f"{base_name}.txt")
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
    
    # Clean up temporary directories
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_frames_dir)
    shutil.rmtree(temp_labels_dir)
    
    return {
        "train": len(train_frames),
        "val": len(val_frames)
    }

def process_video_for_inference(video_path, model, output_path=None, frame_interval=1):
    """
    Process a video with the trained model and create an output video with detections
    
    Args:
        video_path: Path to the input video
        model: Loaded YOLOv8 model
        output_path: Path to save the output video (default: input_detected.mp4)
        frame_interval: Process every Nth frame (for speed)
    
    Returns:
        Path to the output video
    """
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base_name}_detected.mp4"
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    processed_count = 0
    
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_interval == 0:
                # Run inference
                results = model(frame)
                
                # Draw results on frame
                annotated_frame = results[0].plot()
                
                # Write frame
                out.write(annotated_frame)
                processed_count += 1
            else:
                # Write original frame
                out.write(frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Processed {processed_count} frames")
    print(f"Output video saved to {output_path}")
    
    return output_path