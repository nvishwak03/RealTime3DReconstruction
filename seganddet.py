import cv2
import torch
import numpy as np
from ultralytics import YOLO
import warnings

# Suppress warnings
warnings.simplefilter("ignore")

# Load YOLO models
DETECTION_MODEL_PATH = "yolo11n.pt"
SEGMENTATION_MODEL_PATH = "yolo11n-seg.pt"

detection_model = YOLO(DETECTION_MODEL_PATH)
segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Define colors for different object classes
NUM_CLASSES = 80
colors = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)

# Open webcam or video file
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run Detection & Segmentation
    detection_results = detection_model(frame)
    segmentation_results = segmentation_model(frame)

    # Create overlay for segmentation transparency effect
    overlay = frame.copy()

    ## 1️⃣ PROCESS SEGMENTATION RESULTS ##
    for result in segmentation_results:
        masks = result.masks
        classes = result.boxes.cls if result.boxes is not None else []

        if masks is not None:
            for i, mask in enumerate(masks.xy):
                mask = np.array(mask, dtype=np.int32)
                class_id = int(classes[i]) if i < len(classes) else 0
                object_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Object {class_id}"
                color = [int(c) for c in colors[class_id % NUM_CLASSES]]

                # Draw Segmentation Mask
                cv2.fillPoly(overlay, [mask], color=color)
                cv2.polylines(frame, [mask], isClosed=True, color=color, thickness=2)

    # Apply Transparency & Blur Effect for Segmentation Overlay
    alpha = 0.4
    blur = cv2.GaussianBlur(overlay, (15, 15), 5)
    frame = cv2.addWeighted(blur, alpha, frame, 1 - alpha, 0)

    ## 2️⃣ PROCESS DETECTION RESULTS ##
    for result in detection_results:
        boxes = result.boxes.xyxy
        class_ids = result.boxes.cls
        confidences = result.boxes.conf

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(class_ids[i]) if i < len(class_ids) else 0
            confidence = float(confidences[i]) if i < len(confidences) else 0.0
            object_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Object {class_id}"
            color = [int(c) for c in colors[class_id % NUM_CLASSES]]

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw Label with Object Name & Confidence
            label = f"{object_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show final output
    cv2.imshow("YOLO Object Detection & Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
