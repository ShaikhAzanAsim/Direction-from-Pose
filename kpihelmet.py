
# services/kpis/kpi_helmet_detection.py
import numpy as np
from typing import List, Dict, Any
from services.interfaces.ikpi import IKPI
from constants.detections_constant import (
    DETECTION_KEY_BBOX,
    DETECTION_KEY_KEYPOINTS,
    DETECTION_KEY_CLASS_NAME,
    DETECTION_KEY_IN_REGION,
    DETECTION_KEY_REGION_NAME
)


class HelmetDetectionKPI(IKPI):
    """KPI for helmet detection - checks if persons are wearing helmets"""
    
    def __init__(self):
        self.helmet_boxes = []
    
    @staticmethod
    def point_in_box(box, point):
        """Check if a point is inside a bounding box"""
        x1, y1, x2, y2 = box
        px, py = point
        return x1 <= px <= x2 and y1 <= py <= y2
    
    async def calculate(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        tags: List[str] = None,
        snapshot_tags: List[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate helmet detection KPI
        
        Args:
            frame: Input frame
            detections: List of detections containing both helmet and person detections
            tags: List of class names to process (e.g., ['person', 'helmet'])
            snapshot_tags: Tags for snapshot filtering
            
        Returns:
            KPI results with helmet compliance information
        """
        # Separate helmet and person detections
        helmet_detections = [d for d in detections if d.get(DETECTION_KEY_CLASS_NAME) == 'helmet']
        person_detections = [d for d in detections if d.get(DETECTION_KEY_CLASS_NAME) == 'person']
        
        # Extract helmet bounding boxes
        self.helmet_boxes = [d[DETECTION_KEY_BBOX] for d in helmet_detections]
        
        helmet_found = len(self.helmet_boxes) > 0
        
        # Track results
        results = {
            "total_persons": len(person_detections),
            "total_helmets": len(helmet_detections),
            "persons_with_helmet": 0,
            "persons_without_helmet": 0,
            "persons_undetermined": 0,
            "compliance_rate": 0.0,
            "detections_by_region": {},
            "snapshots": []
        }
        
        # Process each person detection
        for person_detection in person_detections:
            helmet_status = self._check_helmet_status(person_detection, helmet_found)
            
            # Update counts
            if helmet_status == "wearing_helmet":
                results["persons_with_helmet"] += 1
            elif helmet_status == "not_wearing_helmet":
                results["persons_without_helmet"] += 1
            else:
                results["persons_undetermined"] += 1
            
            # Store detection info with helmet status
            person_detection["helmet_status"] = helmet_status
            
            # Track by region if available
            if person_detection.get(DETECTION_KEY_IN_REGION):
                region_name = person_detection.get(DETECTION_KEY_REGION_NAME, "global")
                if region_name not in results["detections_by_region"]:
                    results["detections_by_region"][region_name] = {
                        "with_helmet": 0,
                        "without_helmet": 0,
                        "undetermined": 0
                    }
                
                if helmet_status == "wearing_helmet":
                    results["detections_by_region"][region_name]["with_helmet"] += 1
                elif helmet_status == "not_wearing_helmet":
                    results["detections_by_region"][region_name]["without_helmet"] += 1
                else:
                    results["detections_by_region"][region_name]["undetermined"] += 1
        
        # Calculate compliance rate
        if results["total_persons"] > 0:
            results["compliance_rate"] = round(
                (results["persons_with_helmet"] / results["total_persons"]) * 100, 2
            )
        
        return results
    
    def _check_helmet_status(self, person_detection: Dict[str, Any], helmet_found: bool) -> str:
        """
        Check if a person is wearing a helmet
        
        Args:
            person_detection: Person detection dictionary
            helmet_found: Whether any helmets were detected in the frame
            
        Returns:
            Status string: 'wearing_helmet', 'not_wearing_helmet', or 'no_helmet_detected'
        """
        # Get head position from keypoints if available
        if DETECTION_KEY_KEYPOINTS in person_detection and len(person_detection[DETECTION_KEY_KEYPOINTS]) >= 5:
            keypoints = person_detection[DETECTION_KEY_KEYPOINTS]
            # Use first 5 keypoints (nose + eyes + ears) for head position
            head_points = keypoints[0:5]
            
            # Calculate average head position
            valid_points = [kp for kp in head_points if len(kp) >= 3 and kp[2] > 0.3]
            if valid_points:
                head_x = float(np.mean([kp[0] for kp in valid_points]))
                head_y = float(np.mean([kp[1] for kp in valid_points]))
                head_point = (head_x, head_y)
                
                # Check if head is in any helmet box
                helmet_on = any(self.point_in_box(hb, head_point) for hb in self.helmet_boxes)
                
                if helmet_on:
                    return "wearing_helmet"
                elif helmet_found:
                    return "not_wearing_helmet"
                else:
                    return "no_helmet_detected"
        
        # Fallback: use bbox center if keypoints not available
        bbox = person_detection[DETECTION_KEY_BBOX]
        center_x = (bbox[0] + bbox[2]) / 2
        # Use upper third of bbox for head approximation
        center_y = bbox[1] + (bbox[3] - bbox[1]) * 0.25
        head_point = (center_x, center_y)
        
        helmet_on = any(self.point_in_box(hb, head_point) for hb in self.helmet_boxes)
        
        if helmet_on:
            return "wearing_helmet"
        elif helmet_found:
            return "not_wearing_helmet"
        else:
            return "no_helmet_detected"