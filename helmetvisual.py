# services/visualization/helmet_detection_renderer.py
import cv2
import numpy as np
from typing import Dict, Any, List
from services.interfaces.iannotation_renderer import IAnnotationRenderer
from services.managers.color_manager import ColorManager
from dtos.request.request_register_use_case import Region
from constants.detections_constant import (
    DETECTION_KEY_BBOX,
    DETECTION_KEY_CLASS_NAME,
    DETECTION_KEY_KEYPOINTS,
    DETECTION_KEY_CONFIDENCE
)


class HelmetDetectionRenderer(IAnnotationRenderer):
    """Renderer for helmet detection visualization"""
    
    def render(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        regions: List[Region],
        color_manager: ColorManager,
        **kwargs
    ) -> np.ndarray:
        """
        Render helmet detection annotations on frame
        
        Args:
            frame: Input frame
            detection: Detection dictionary
            regions: List of regions
            color_manager: Color manager for consistent colors
            **kwargs: Additional parameters
            
        Returns:
            Annotated frame
        """
        class_name = detection.get(DETECTION_KEY_CLASS_NAME, "unknown")
        bbox = detection.get(DETECTION_KEY_BBOX, [])
        
        if len(bbox) < 4:
            return frame
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Determine color based on class and helmet status
        if class_name == "helmet":
            color = (0, 255, 0)  # Green for helmet
            label = "Helmet"
        elif class_name == "person":
            helmet_status = detection.get("helmet_status", "unknown")
            
            if helmet_status == "wearing_helmet":
                color = (0, 255, 0)  # Green
                label = "Person - Helmet ON"
            elif helmet_status == "not_wearing_helmet":
                color = (0, 0, 255)  # Red
                label = "Person - Helmet OFF"
            else:
                color = (0, 255, 255)  # Yellow
                label = "Person - No Helmet"
        else:
            color = (255, 255, 255)
            label = class_name
        
        # Add confidence if available
        confidence = detection.get(DETECTION_KEY_CONFIDENCE, 0)
        if confidence > 0:
            label += f" {confidence:.2f}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        
        # Draw head keypoint if available
        if DETECTION_KEY_KEYPOINTS in detection and class_name == "person":
            keypoints = detection[DETECTION_KEY_KEYPOINTS]
            if len(keypoints) >= 5:
                head_points = keypoints[0:5]
                valid_points = [kp for kp in head_points if len(kp) >= 3 and kp[2] > 0.3]
                
                if valid_points:
                    head_x = int(np.mean([kp[0] for kp in valid_points]))
                    head_y = int(np.mean([kp[1] for kp in valid_points]))
                    cv2.circle(frame, (head_x, head_y), 5, (255, 0, 0), -1)
        
        return frame