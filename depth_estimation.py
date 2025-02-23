import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLO Models
DETECTION_MODEL_PATH = "yolo11n.pt"
SEGMENTATION_MODEL_PATH = "yolo11n-seg.pt"

detection_model = YOLO(DETECTION_MODEL_PATH)
segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)

# Load MiDaS Depth Estimation Model
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # Use "MiDaS_large" for better quality
midas_model.eval()

# Move MiDaS model to GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas_model.to(DEVICE)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Define colors for different object classes
NUM_CLASSES = 80
colors = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)

# Open webcam or video file
cap = cv2.VideoCapture(0)  # Change to "video.mp4" for file input

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run Detection & Segmentation
    detection_results = detection_model(frame)
    segmentation_results = segmentation_model(frame)

    # Convert frame to RGB for MiDaS
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_input = midas_transforms(img_rgb).unsqueeze(0).to(DEVICE)  # Ensure it has shape (1, 3, H, W)

    # Verify Tensor Shape
    print(f"Shape of img_input: {img_input.shape}")

    # Predict Depth using MiDaS
    with torch.no_grad():
        depth_map = midas_model(img_input)

    # Normalize Depth Map
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Resize Depth Map to Match Frame Size
    depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # Convert Depth to Color Map
    depth_colormap = cv2.applyColorMap((depth_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Create overlay for segmentation transparency
    overlay = frame.copy()

    ## 1️⃣ PROCESS SEGMENTATION RESULTS ##
    for result in segmentation_results:
        masks = result.masks
        classes = result.boxes.cls if result.boxes is not None else []

        if masks is not None:
            for i, mask in enumerate(masks.xy):
                mask = np.array(mask, dtype=np.int32)
                class_id = int(classes[i]) if i < len(classes) else 0
                color = [int(c) for c in colors[class_id % NUM_CLASSES]]

                # Draw Segmentation Mask
                cv2.fillPoly(overlay, [mask], color=color)
                cv2.polylines(frame, [mask], isClosed=True, color=color, thickness=2)

    # Apply Blur & Transparency Effect for Segmentation
    alpha = 0.4
    blur = cv2.GaussianBlur(overlay, (15, 15), 5)
    frame = cv2.addWeighted(blur, alpha, frame, 1 - alpha, 0)

    ## 2️⃣ PROCESS DETECTION RESULTS ##
    for result in detection_results:
        boxes = result.boxes.xyxy  # Bounding Boxes
        class_ids = result.boxes.cls  # Class Labels
        confidences = result.boxes.conf  # Confidence Scores

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Convert box to int
            class_id = int(class_ids[i]) if i < len(class_ids) else 0
            confidence = float(confidences[i]) if i < len(confidences) else 0.0
            color = [int(c) for c in colors[class_id % NUM_CLASSES]]

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw Label
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Overlay Depth Value on Each Object
            depth_value = depth_map_resized[y1:y2, x1:x2].mean()
            depth_text = f"Depth: {depth_value:.2f}"
            cv2.putText(frame, depth_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Overlay Depth Map on Side for Visualization
    blended = np.hstack((frame, depth_colormap))

    # Show Final Output
    cv2.imshow("YOLO + Segmentation + Depth Estimation", blended)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
