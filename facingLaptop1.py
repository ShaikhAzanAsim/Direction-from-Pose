import cv2
import numpy as np
from ultralytics import YOLO

# Load models
pose_model = YOLO("yolov8n-pose.pt")      # For person keypoints
object_model = YOLO("yolov8n.pt")         # For laptop detection

# Label for laptop in COCO dataset
LAPTOP_CLASS_ID = 63   # In COCO, 'laptop' = 63


def get_center_of_box(box):
    """Get center (x, y) of a YOLO box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def compute_angle_between_vectors(v1, v2):
    """Compute the angle in degrees between two 2D vectors."""
    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if mag1 == 0 or mag2 == 0:
        return 180.0
    cos_theta = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))
    return theta


def is_facing_laptop(person_kps, laptop_centers, angle_thresh=30):
    """
    Determine if a person (keypoints) is facing any laptop in the frame.
    """
    nose = person_kps[0]
    left_shoulder = person_kps[5]
    right_shoulder = person_kps[6]

    # Check confidence
    if nose[2] < 0.5 or left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5:
        return False, None, None

    # Person center (midpoint of shoulders)
    C_person = np.array([
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    ])

    # Facing vector (person center → nose)
    V_facing = np.array([nose[0] - C_person[0], nose[1] - C_person[1]])

    best_angle = 999
    best_laptop = None

    for laptop_center in laptop_centers:
        V_target = np.array([laptop_center[0] - C_person[0], laptop_center[1] - C_person[1]])
        theta = compute_angle_between_vectors(V_facing, V_target)

        if theta < best_angle:
            best_angle = theta
            best_laptop = laptop_center

    if best_angle < angle_thresh:
        return True, best_laptop, best_angle
    else:
        return False, best_laptop, best_angle


# ------------------ MAIN LOOP ------------------
video_path = r"C:\Users\QBS PC\QBS_CO\DEMO\FacingDirection\CCTV_Office_Scene_Generation.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection models
    obj_results = object_model(frame, verbose=False)
    pose_results = pose_model(frame, verbose=False)

    # Get all laptop centers
    laptop_centers = []
    for r in obj_results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) == LAPTOP_CLASS_ID:
                laptop_centers.append(get_center_of_box(box.cpu().numpy()))

    # Draw laptop detections
    for c in laptop_centers:
        cv2.circle(frame, c, 8, (255, 255, 0), -1)
        cv2.putText(frame, "Laptop", (c[0] - 30, c[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Process each detected person
    for r in pose_results:
        for kp in r.keypoints.data:
            kps = kp.cpu().numpy().reshape(-1, 3)

            facing, best_laptop, angle = is_facing_laptop(kps, laptop_centers)

            # Draw nose and shoulders
            for (x, y, conf) in kps[[0, 5, 6]]:
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            # Draw facing vector
            nose = kps[0]
            left_shoulder = kps[5]
            right_shoulder = kps[6]
            Cx = (left_shoulder[0] + right_shoulder[0]) / 2
            Cy = (left_shoulder[1] + right_shoulder[1]) / 2
            cv2.arrowedLine(frame, (int(Cx), int(Cy)), (int(nose[0]), int(nose[1])),
                            (255, 0, 0), 2, tipLength=0.3)

            # Decision text
            text_pos = (int(Cx), int(Cy) - 30)
            if facing:
                cv2.putText(frame, f"Facing Laptop ({angle:.1f})", text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Draw connecting line
                if best_laptop is not None:
                    cv2.line(frame, (int(Cx), int(Cy)), best_laptop, (0, 255, 0), 2)
            else:
                status = "Not Facing" if laptop_centers else "No Laptops"
                angle_text = f" ({angle:.1f})" if angle is not None else ""
                cv2.putText(frame, f"{status}{angle_text}", text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Person Facing Laptop Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
