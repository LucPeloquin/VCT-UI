"""
Script for processing videos for Valorant UI detection
"""

import os
import argparse
from video_processor import extract_frames, apply_static_annotations, prepare_video_dataset, process_video_for_inference
from data.dataset import create_dataset_yaml
from models.detector import setup_model
import config

def parse_args():
    parser = argparse.ArgumentParser(description="Process video for Valorant UI detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract frames command
    extract_parser = subparsers.add_parser("extract", help="Extract frames from video")
    extract_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    extract_parser.add_argument("--output", type=str, default="frames", help="Output directory for frames")
    extract_parser.add_argument("--interval", type=int, default=30, help="Extract every Nth frame")
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset from video with static annotations")
    prepare_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    prepare_parser.add_argument("--dataset", type=str, default=config.DATASET_DIR, help="Dataset directory")
    prepare_parser.add_argument("--template", type=str, required=True, help="Path to template annotation file")
    prepare_parser.add_argument("--interval", type=int, default=30, help="Extract every Nth frame")
    
    # Process video command
    process_parser = subparsers.add_parser("process", help="Process video with trained model")
    process_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    process_parser.add_argument("--weights", type=str, default=config.BEST_WEIGHTS, help="Path to model weights")
    process_parser.add_argument("--output", type=str, default=None, help="Output video path")
    process_parser.add_argument("--interval", type=int, default=1, help="Process every Nth frame")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.command == "extract":
        # Extract frames from video
        extract_frames(args.video, args.output, args.interval)
        print(f"Now you can annotate a single frame (e.g., frames/frame_00000.png) using LabelImg")
        print(f"Then use the 'prepare' command with the annotation file to create a dataset")
    
    elif args.command == "prepare":
        # Prepare dataset from video with static annotations
        stats = prepare_video_dataset(
            args.video, 
            args.dataset, 
            args.template, 
            args.interval
        )
        
        print(f"Dataset prepared with {stats['train']} training and {stats['val']} validation images")
        
        # Create dataset YAML
        yaml_path = create_dataset_yaml(args.dataset, config.CLASSES)
        print(f"Created dataset YAML: {yaml_path}")
        print(f"You can now train your model with: python train.py --skip-collect")
    
    elif args.command == "process":
        # Load model
        model = setup_model(args.weights)
        
        # Process video
        output_path = process_video_for_inference(
            args.video,
            model,
            args.output,
            args.interval
        )
        
        print(f"Processed video saved to: {output_path}")
    
    else:
        print("Please specify a command: extract, prepare, or process")
        print("Run with --help for more information")

if __name__ == "__main__":
    main()