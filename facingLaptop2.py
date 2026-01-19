import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------
# CONFIGURATION
# ---------------------------
THRESHOLD_ANGLE = 30   # degrees: adjust to test sensitivity
MIN_CONFIDENCE = 0.4   # keypoint minimum confidence
HARDWARE_CLASSES = {62, 63, 64, 65, 66}  # tv, laptop, mouse, remote, keyboard

# ---------------------------
# LOAD MODELS
# ---------------------------
yolo_obj = YOLO("yolov8n.pt")         # object detector (hardware)
yolo_pose = YOLO("yolov8n-pose.pt")   # pose detector (people)

# ---------------------------
# HELPERS
# ---------------------------
def get_center(box):
    x1, y1, x2, y2 = box[:4]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_person_center_and_nose(keypoints):
    """Return (shoulder midpoint, nose) given full keypoint array [17,3]."""
    if keypoints.shape[0] < 7:
        return None, None

    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    if left_shoulder[2] < MIN_CONFIDENCE or right_shoulder[2] < MIN_CONFIDENCE:
        return None, None

    cx = int((left_shoulder[0] + right_shoulder[0]) / 2)
    cy = int((left_shoulder[1] + right_shoulder[1]) / 2)
    if nose[2] < MIN_CONFIDENCE:
        # If nose low confidence, still return shoulders center
        return (cx, cy), None
    return (cx, cy), (int(nose[0]), int(nose[1]))

def calculate_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return None
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# ---------------------------
# PROCESS VIDEO
# ---------------------------
video_path = r"C:\Users\QBS PC\QBS_CO\DEMO\FacingDirection\CCTV_Office_Scene_Generation.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Detect hardware (laptops, tvs, etc.)
    obj_results = yolo_obj(frame, verbose=False)[0]
    hardware_boxes = [
        box.xyxy[0].cpu().numpy()
        for box in obj_results.boxes
        if int(box.cls[0]) in HARDWARE_CLASSES
    ]

    # 2️⃣ Detect people (pose)
    pose_results = yolo_pose(frame, verbose=False)[0]
    keypoints_all = pose_results.keypoints.data.cpu().numpy() if pose_results.keypoints is not None else []

    for person_kp in keypoints_all:
        kps = person_kp.reshape(-1, 3)
        C_person, C_nose = get_person_center_and_nose(kps)
        if C_person is None:
            continue

        # Draw shoulders midpoint
        cv2.circle(frame, C_person, 6, (255, 255, 0), -1)

        # Draw shoulders themselves if visible
        for idx in [5, 6]:
            if kps[idx][2] > MIN_CONFIDENCE:
                cv2.circle(frame, (int(kps[idx][0]), int(kps[idx][1])), 5, (0, 165, 255), -1)

        # Draw nose if visible
        if C_nose is not None:
            cv2.circle(frame, C_nose, 5, (0, 255, 0), -1)
            cv2.line(frame, C_person, C_nose, (0, 255, 255), 2)  # face vector (always drawn)
        else:
            cv2.putText(frame, "No Nose", (C_person[0] - 40, C_person[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Face vector
        if C_nose is not None:
            V_facing = np.array([C_nose[0] - C_person[0], C_nose[1] - C_person[1]])
        else:
            continue  # skip angle computation if no nose

        closest_hw = None
        min_angle = None

        for hw_box in hardware_boxes:
            C_hw = get_center(hw_box)
            cv2.rectangle(frame, (int(hw_box[0]), int(hw_box[1])),
                          (int(hw_box[2]), int(hw_box[3])), (255, 0, 0), 2)
            cv2.circle(frame, C_hw, 4, (255, 0, 255), -1)

            V_target = np.array([C_hw[0] - C_person[0], C_hw[1] - C_person[1]])
            angle = calculate_angle(V_facing, V_target)

            if angle is not None and (min_angle is None or angle < min_angle):
                min_angle = angle
                closest_hw = C_hw

        # Decision + draw target vector
        if closest_hw is not None and min_angle is not None:
            color = (0, 255, 0) if min_angle < THRESHOLD_ANGLE else (0, 0, 255)
            status = "Facing HW" if min_angle < THRESHOLD_ANGLE else "Not Facing"
            cv2.line(frame, C_person, closest_hw, (255, 0, 255), 2)
            cv2.putText(frame, f"{status} ({min_angle:.1f}°)",
                        (C_person[0] - 60, C_person[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "No Hardware", (C_person[0] - 50, C_person[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    cv2.putText(frame, f"THRESHOLD={THRESHOLD_ANGLE}°", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Facing Direction Debug View", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
