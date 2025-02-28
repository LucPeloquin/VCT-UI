"""
Inference and post-processing for Valorant UI detection
"""

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

class ValorantUIProcessor:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """Initialize the Valorant UI processor"""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def detect(self, image_path):
        """Run detection on an image"""
        results = self.model(image_path, conf=self.conf_threshold, iou=self.iou_threshold)
        return results[0]  # Return first result
    
    def process_image(self, image_path):
        """Process image and extract UI information"""
        # Run detection
        results = self.detect(image_path)
        
        # Load original image
        img = cv2.imread(image_path)
        
        # Dictionary to store extracted information
        ui_info = {
            'score': None,
            'timer': None,
            'teams': [],
            'players': [],
            'round': None,
            'map_name': None
        }
        
        # Process each detection
        for i in range(len(results.boxes)):
            box = results.boxes[i]
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Extract ROI
            roi = img[y1:y2, x1:x2]
            
            # Process based on class
            if cls_name == 'score_display':
                ui_info['score'] = self._extract_score(roi)
            elif cls_name == 'timer':
                ui_info['timer'] = self._extract_timer(roi)
            elif cls_name == 'team_logo':
                team_info = self._process_team_logo(roi, x1, img.shape[1])
                ui_info['teams'].append(team_info)
            elif cls_name == 'player_panel':
                player_info = self._process_player_panel(roi)
                ui_info['players'].append(player_info)
            elif cls_name == 'round_indicator':
                ui_info['round'] = self._extract_round_info(roi)
        
        return ui_info, results
    
    def _extract_score(self, roi):
        """Extract score from ROI using OCR"""
        # Preprocess for better OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # OCR
        score_text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789-')
        return score_text.strip()
    
    def _extract_timer(self, roi):
        """Extract timer from ROI using OCR"""
        # Preprocess for better OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # OCR
        timer_text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789:')
        return timer_text.strip()
    
    def _process_team_logo(self, roi, x_pos, img_width):
        """Process team logo ROI"""
        # Determine if team is on left or right side
        side = 'left' if x_pos < img_width/2 else 'right'
        
        # For now, just store the ROI and position
        # In a real implementation, you might use template matching to identify teams
        return {
            'side': side,
            'logo_roi': roi
        }
    
    def _process_player_panel(self, roi):
        """Process player panel ROI to extract player information"""
        h, w = roi.shape[:2]
        
        # Extract regions (these would need to be adjusted based on actual UI)
        agent_roi = roi[0:h, 0:int(h*0.8)]  # Square region for agent icon
        name_roi = roi[0:int(h/2), int(h*0.8):int(w*0.6)]  # Region for player name
        health_roi = roi[int(h/2):h, int(h*0.8):int(w*0.6)]  # Region for health/shield
        weapon_roi = roi[0:h, int(w*0.6):w]  # Region for weapon
        
        # OCR for player name
        gray = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        player_name = pytesseract.image_to_string(thresh).strip()
        
        return {
            'name': player_name,
            'agent_roi': agent_roi,
            'health_roi': health_roi,
            'weapon_roi': weapon_roi
        }
    
    def _extract_round_info(self, roi):
        """Extract round information from ROI using OCR"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        round_text = pytesseract.image_to_string(thresh)
        return round_text.strip() 