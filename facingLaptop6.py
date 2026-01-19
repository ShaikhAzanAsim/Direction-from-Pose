import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------
# CONFIGURATION
# ---------------------------
THRESHOLD_ANGLE = 30      # degrees: facing threshold
DISTANCE_THRESHOLD = 250  # pixels: proximity threshold
MIN_CONFIDENCE = 0.4
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

def rotate_vector_90_deg(vec):
    """Rotate a 2D vector by +90 degrees (clockwise)."""
    x, y = vec
    return np.array([y, -x])

def get_person_center_and_facing(keypoints):
    """
    Return (shoulder midpoint, facing point)
    Facing direction is estimated purely from shoulders.
    """
    if keypoints.shape[0] < 7:
        return None, None

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    if left_shoulder[2] < MIN_CONFIDENCE or right_shoulder[2] < MIN_CONFIDENCE:
        return None, None

    # Midpoint between shoulders
    cx = int((left_shoulder[0] + right_shoulder[0]) / 2)
    cy = int((left_shoulder[1] + right_shoulder[1]) / 2)
    C_person = (cx, cy)

    # Shoulder line vector (left → right)
    shoulder_vec = np.array([
        left_shoulder[0] - right_shoulder[0],
        left_shoulder[1] - right_shoulder[1]
    ])

    if np.linalg.norm(shoulder_vec) == 0:
        return C_person, None

    # Normalize
    shoulder_vec = shoulder_vec / np.linalg.norm(shoulder_vec)

    # Rotate 90 degrees (to estimate front direction)
    rotated_vec = rotate_vector_90_deg(shoulder_vec)

    # 🔁 Reverse (depends on camera view — this flips to correct facing)
    rotated_vec = -rotated_vec

    # Estimate facing point 50px ahead
    face_x = int(C_person[0] + 50 * rotated_vec[0])
    face_y = int(C_person[1] + 50 * rotated_vec[1])
    return C_person, (face_x, face_y)

def calculate_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return None
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# ---------------------------
# PROCESS VIDEO
# ---------------------------
video_path = r"C:\Users\QBS PC\QBS_CO\qbs_camera\passByTraffic\passBy1Trimmed.mp4"
cap = cv2.VideoCapture(video_path)
out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Detect hardware
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
        C_person, C_face = get_person_center_and_facing(kps)
        if C_person is None or C_face is None:
            continue

        # Draw shoulders midpoint
        cv2.circle(frame, C_person, 6, (255, 255, 0), -1)

        # Draw shoulder keypoints
        for idx in [5, 6]:
            if kps[idx][2] > MIN_CONFIDENCE:
                cv2.circle(frame, (int(kps[idx][0]), int(kps[idx][1])), 5, (0, 165, 255), -1)

        # Draw facing direction arrow
        V_facing = np.array([C_face[0] - C_person[0], C_face[1] - C_person[1]])
        arrow_len = 80
        if np.linalg.norm(V_facing) != 0:
            unit_vec = V_facing / np.linalg.norm(V_facing)
            arrow_tip = (int(C_face[0] + arrow_len * unit_vec[0]),
                         int(C_face[1] + arrow_len * unit_vec[1]))
            cv2.arrowedLine(frame, C_face, arrow_tip, (0, 255, 255), 3, tipLength=0.3)
            cv2.putText(frame, "Facing (shoulders)", (C_person[0]-70, C_person[1]-70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        closest_hw = None
        min_angle = None
        min_distance = None

        # Find nearest hardware
        for hw_box in hardware_boxes:
            C_hw = get_center(hw_box)
            cv2.rectangle(frame, (int(hw_box[0]), int(hw_box[1])),
                          (int(hw_box[2]), int(hw_box[3])), (255, 0, 0), 2)
            cv2.circle(frame, C_hw, 4, (255, 0, 255), -1)

            V_target = np.array([C_hw[0] - C_person[0], C_hw[1] - C_person[1]])
            angle = calculate_angle(V_facing, V_target)
            distance = np.linalg.norm(V_target)

            if angle is not None and (min_angle is None or angle < min_angle):
                min_angle = angle
                closest_hw = C_hw
                min_distance = distance

        # Determine working / not working
        if closest_hw is not None and min_angle is not None:
            working = (min_angle < THRESHOLD_ANGLE and min_distance < DISTANCE_THRESHOLD)
            color = (0, 255, 0) if working else (0, 0, 255)
            status = "Working" if working else "Not Working"

            cv2.line(frame, C_person, closest_hw, (255, 0, 255), 2)
            cv2.putText(frame, f"{status} ({min_angle:.1f}°, d={int(min_distance)})",
                        (C_person[0] - 80, C_person[1] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "No Hardware", (C_person[0] - 50, C_person[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    # Display thresholds
    cv2.putText(frame, f"TH_ANGLE={THRESHOLD_ANGLE}°, TH_DIST={DISTANCE_THRESHOLD}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Initialize writer
    if out is None:
        out = cv2.VideoWriter('output_facing_shoulders3.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              20.0, (frame.shape[1], frame.shape[0]))
    out.write(frame)

    cv2.imshow("Facing Direction (Shoulders Only)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release
if out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()
