Valorant UI Detector
A custom object detection system for analyzing Valorant game UI elements using YOLOv8.
Setup
Install dependencies:
txt
Prepare your dataset:
Collect Valorant screenshots in a folder (e.g., valorant_screenshots)
Annotate the images using tools like LabelImg or CVAT
Save annotations in YOLO format
Train the model:
augment
Run detection:
png
Project Structure
data/: Dataset preparation and augmentation
models/: YOLOv8 model setup and training
utils/: Visualization utilities
inference/: Inference and post-processing
config.py: Configuration settings
train.py: Training script
detect.py: Inference script
UI Elements Detected
Score display
Timer
Team logos
Player panels
Weapon icons
Agent icons
Minimap
Round indicator
Health/shield bars
Ability icons
Example Usage
)
Output Format
The detector outputs both a visualization image and a JSON file containing structured information about the UI elements:
}
Command Line Arguments
Training
--source: Directory with source screenshots
--dataset: Dataset directory
--epochs: Number of training epochs
--batch: Batch size
--img-size: Image size
--model: YOLOv8 model type
--augment: Apply data augmentation
--skip-collect: Skip data collection (use existing dataset)
Detection
--image: Path to input image
--weights: Path to model weights
--conf: Confidence threshold
--iou: IoU threshold
--output: Path to output image
--json: Path to output JSON