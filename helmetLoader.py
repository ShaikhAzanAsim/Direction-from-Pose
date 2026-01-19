# services/loaders/helmet_detection_data_loader.py
import numpy as np
from typing import List, Dict, Any
from services.interfaces.idata_loader import IDataLoader
from constants.detections_constant import (
    DETECTION_KEY_BBOX,
    DETECTION_KEY_CLASS_NAME,
    DETECTION_KEY_CONFIDENCE,
    DETECTION_KEY_KEYPOINTS
)


class HelmetDetectionDataLoader(IDataLoader):
    """Data loader for helmet detection models (combined helmet + pose)"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def load(self, result, frame: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Load detections from YOLO results for helmet detection
        
        Args:
            result: YOLO result object
            frame: Optional frame for additional processing
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Process bounding boxes (helmets and persons)
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                detection = {
                    DETECTION_KEY_BBOX: box.tolist(),
                    DETECTION_KEY_CONFIDENCE: float(conf),
                    DETECTION_KEY_CLASS_NAME: result.names[class_id]
                }
                detections.append(detection)
        
        # Process keypoints if available (for pose estimation)
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints_data = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy() if hasattr(result.keypoints, 'conf') else None
            
            # Match keypoints with person detections
            person_idx = 0
            for detection in detections:
                if detection[DETECTION_KEY_CLASS_NAME] == 'person' and person_idx < len(keypoints_data):
                    kps = keypoints_data[person_idx]
                    
                    if keypoints_conf is not None:
                        kps_conf = keypoints_conf[person_idx]
                        # Combine x, y, confidence
                        keypoints = [[float(kps[j][0]), float(kps[j][1]), float(kps_conf[j])] 
                                   for j in range(len(kps))]
                    else:
                        keypoints = [[float(kps[j][0]), float(kps[j][1]), 1.0] 
                                   for j in range(len(kps))]
                    
                    detection[DETECTION_KEY_KEYPOINTS] = keypoints
                    person_idx += 1
        
        return detections