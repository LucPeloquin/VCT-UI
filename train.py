"""
Training script for Valorant UI detector
"""

import os
import argparse
from data.dataset import setup_dataset_directories, collect_screenshots, create_dataset_yaml, check_annotations
from data.augmentation import basic_augmentation
from models.detector import setup_model, train_model, export_model
from utils.visualization import plot_training_results
import config

def parse_args():
    parser = argparse.ArgumentParser(description="Train Valorant UI detector")
    parser.add_argument("--source", type=str, default=config.SOURCE_DIR, help="Directory with source screenshots")
    parser.add_argument("--dataset", type=str, default=config.DATASET_DIR, help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE, help="Image size")
    parser.add_argument("--model", type=str, default=config.MODEL_TYPE, help="YOLOv8 model type")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--skip-collect", action="store_true", help="Skip data collection (use existing dataset)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup dataset directories
    dataset_dir = setup_dataset_directories(args.dataset)
    print(f"Dataset directory: {dataset_dir}")
    
    # Collect screenshots
    if not args.skip_collect:
        n_train, n_val = collect_screenshots(args.source, dataset_dir, config.TRAIN_VAL_SPLIT)
        print(f"Collected {n_train} training and {n_val} validation images")
    
    # Check if annotations exist
    annotations_exist = check_annotations(dataset_dir)
    if not annotations_exist:
        print("Warning: Some images are missing annotations.")
        print("Please annotate your images before proceeding.")
        print("You can use tools like LabelImg or CVAT for annotation.")
        return
    
    # Apply data augmentation if requested
    if args.augment:
        n_train_aug = basic_augmentation(dataset_dir)
        print(f"Applied augmentation, now have {n_train_aug} training images")
    
    # Create dataset YAML
    yaml_path = create_dataset_yaml(dataset_dir, config.CLASSES)
    print(f"Created dataset YAML: {yaml_path}")
    
    # Setup model
    model = setup_model(args.model)
    
    # Train model
    results = train_model(
        model, 
        yaml_path, 
        epochs=args.epochs, 
        img_size=args.img_size, 
        batch_size=args.batch
    )
    
    # Export model
    export_model(model, format='onnx')
    
    # Plot training results
    results_file = os.path.join(config.OUTPUT_DIR, "detect", "valorant_ui_detector", "results.csv")
    plot_training_results(results_file)
    
    print("Training complete!")
    print(f"Best weights saved to: {config.BEST_WEIGHTS}")

if __name__ == "__main__":
    main() 