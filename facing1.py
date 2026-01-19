import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")  # or yolov8s-pose.pt for more accuracy

def get_facing_direction(keypoints):
    """
    Estimate facing direction ('left', 'right', 'up', 'down') from pose keypoints.
    """
    if keypoints is None or len(keypoints) < 7:
        return "unknown"

    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    # Confidence check
    if nose[2] < 0.5 or left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5:
        return "back"

    # Center of shoulders
    Cx = (left_shoulder[0] + right_shoulder[0]) / 2
    Cy = (left_shoulder[1] + right_shoulder[1]) / 2

    # Facing vector
    Vx = nose[0] - Cx
    Vy = nose[1] - Cy

    angle = np.degrees(np.arctan2(Vy, Vx))

    # Optional: use shoulder slope to refine top-down view
    shoulder_angle = np.degrees(np.arctan2(
        right_shoulder[1] - left_shoulder[1],
        right_shoulder[0] - left_shoulder[0]
    ))

    # Determine direction
    if abs(Vx) < 5 and abs(Vy) < 5:
        # Probably top-down overlap — use shoulders instead
        if shoulder_angle > 10:
            return "right"
        elif shoulder_angle < -10:
            return "left"
        else:
            return "front/back unclear"

    if -45 <= angle <= 45:
        return "right"
    elif 45 < angle <= 135:
        return "down"
    elif -135 <= angle < -45:
        return "up"
    else:
        return "left"


# ---------- Run on sample CCTV footage ----------
video_path = r"C:\Users\QBS PC\QBS_CO\DEMO\FacingDirection\CCTV_Office_Scene_Generation.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 Pose inference
    results = model(frame, verbose=False)

    for r in results:
        for kp in r.keypoints.data:
            kps = kp.cpu().numpy().reshape(-1, 3)
            direction = get_facing_direction(kps)

            # Draw nose and shoulders
            for (x, y, conf) in kps[[0, 5, 6]]:
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            # Draw text
            nose_x, nose_y, _ = kps[0]
            cv2.putText(frame, direction, (int(nose_x), int(nose_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Facing Direction Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
