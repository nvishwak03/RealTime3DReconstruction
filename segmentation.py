import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLO segmentation model
MODEL_PATH = "yolo11n-seg.pt"  # Update with your model path
model = YOLO(MODEL_PATH)

# Define colors for different object classes (randomly generated)
NUM_CLASSES = 80  # Change this based on your dataset
colors = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)

# Open webcam or video file
cap = cv2.VideoCapture(0)  # Change to "video.mp4" for file input

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Create an overlay for semi-transparent effect
    overlay = frame.copy()

    for result in results:
        masks = result.masks
        classes = result.boxes.cls if result.boxes is not None else []

        if masks is not None:
            for i, mask in enumerate(masks.xy):
                mask = np.array(mask, dtype=np.int32)
                class_id = int(classes[i]) if i < len(classes) else 0  # Get class ID
                color = [int(c) for c in colors[class_id % NUM_CLASSES]]  # Assign color
                
                # Draw mask with light transparency
                cv2.fillPoly(overlay, [mask], color=color)
                
                # Draw boundary
                cv2.polylines(frame, [mask], isClosed=True, color=color, thickness=2)

    # Apply transparent overlay with slight blur effect
    alpha = 0.4  # Transparency factor
    blur = cv2.GaussianBlur(overlay, (15, 15), 5)
    frame = cv2.addWeighted(blur, alpha, frame, 1 - alpha, 0)

    # Display output
    cv2.imshow("YOLO Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
