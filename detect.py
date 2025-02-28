"""
Inference script for Valorant UI detector
"""

import os
import argparse
import cv2
from inference.processor import ValorantUIProcessor
from utils.visualization import draw_detections
import config
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Run Valorant UI detection")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default=config.BEST_WEIGHTS, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD, help="IoU threshold")
    parser.add_argument("--output", type=str, default="output.png", help="Path to output image")
    parser.add_argument("--json", type=str, default="output.json", help="Path to output JSON")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    # Check if weights exist
    if not os.path.exists(args.weights):
        print(f"Error: Weights not found at {args.weights}")
        return
    
    # Initialize processor
    processor = ValorantUIProcessor(args.weights, args.conf, args.iou)
    
    # Process image
    ui_info, results = processor.process_image(args.image)
    
    # Draw detections
    img = cv2.imread(args.image)
    output_img = draw_detections(img, results.boxes, results.names, args.conf)
    
    # Save output image
    cv2.imwrite(args.output, output_img)
    print(f"Detection visualization saved to {args.output}")
    
    # Save UI information to JSON
    # Convert numpy arrays to lists for JSON serialization
    for team in ui_info['teams']:
        if 'logo_roi' in team:
            team['logo_roi'] = team['logo_roi'].tolist() if hasattr(team['logo_roi'], 'tolist') else None
    
    for player in ui_info['players']:
        if 'agent_roi' in player:
            player['agent_roi'] = player['agent_roi'].tolist() if hasattr(player['agent_roi'], 'tolist') else None
        if 'health_roi' in player:
            player['health_roi'] = player['health_roi'].tolist() if hasattr(player['health_roi'], 'tolist') else None
        if 'weapon_roi' in player:
            player['weapon_roi'] = player['weapon_roi'].tolist() if hasattr(player['weapon_roi'], 'tolist') else None
    
    with open(args.json, 'w') as f:
        json.dump(ui_info, f, indent=4)
    
    print(f"UI information saved to {args.json}")
    
    # Print summary
    print("\nDetection Summary:")
    print(f"Score: {ui_info['score']}")
    print(f"Timer: {ui_info['timer']}")
    print(f"Round: {ui_info['round']}")
    print(f"Teams: {len(ui_info['teams'])}")
    print(f"Players: {len(ui_info['players'])}")
    
    # Print player names if available
    if ui_info['players']:
        print("\nDetected Players:")
        for i, player in enumerate(ui_info['players']):
            print(f"  {i+1}. {player['name']}")

if __name__ == "__main__":
    main() 