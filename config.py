"""
Configuration settings for Valorant UI detector
"""

import os

# Dataset settings
DATASET_DIR = "valorant_dataset"
SOURCE_DIR = "valorant_screenshots"  # Directory with raw screenshots
TRAIN_VAL_SPLIT = 0.8

# Model settings
MODEL_TYPE = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 50

# Classes for detection
CLASSES = [
    'score_display',
    'timer',
    'team_logo',
    'player_panel',
    'weapon_icon',
    'agent_icon',
    'minimap',
    'round_indicator',
    'health_bar',
    'ability_icon'
]

# Inference settings
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Paths
OUTPUT_DIR = "runs"
WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "detect", "valorant_ui_detector", "weights")
BEST_WEIGHTS = os.path.join(WEIGHTS_DIR, "best.pt") 