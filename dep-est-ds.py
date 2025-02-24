import cv2
import numpy as np
import torch
import open3d as o3d
from ultralytics import YOLO

# Paths to your YOLO11 models
DETECTION_MODEL_PATH = "yolo11n.pt"
SEGMENTATION_MODEL_PATH = "yolo11n-seg.pt"

# Load YOLO11 models
detection_model = YOLO(DETECTION_MODEL_PATH)
segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)

# Load MiDaS Small model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Initialize Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Point Cloud', width=800, height=600)
pcd = o3d.geometry.PointCloud()
added = False

def estimate_depth(image, model_type="detection"):
    """
    Estimate depth for detected or segmented objects in the image.
    model_type: 'detection' for bounding boxes, 'segmentation' for masks
    """
    # Select the appropriate YOLO model
    model = detection_model if model_type == "detection" else segmentation_model
    
    # Perform inference with YOLO
    results = model(image, verbose=False)[0]  # Suppress verbose output for speed
    
    # Convert image to RGB for MiDaS
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess image for MiDaS
    input_batch = transform(img_rgb).to(device)
    
    # Compute depth map with MiDaS
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert depth map to numpy
    depth_map = prediction.cpu().numpy()
    
    # Normalize depth map for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)  # Changed to JET
    
    # Process YOLO results and overlay depth information
    output_image = image.copy()
    
    if model_type == "detection":
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{results.names[cls]} {conf:.2f}"
            
            # Calculate average depth in the bounding box region
            depth_region = depth_map[y1:y2, x1:x2]
            avg_depth = np.mean(depth_region)
            
            # Draw bounding box and label
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_image, f"{label} Depth: {avg_depth:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    elif model_type == "segmentation":
        if results.masks is not None:
            for mask in results.masks.data:
                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask = mask > 0.5  # Binary mask
                
                # Calculate average depth in the segmented region
                depth_region = depth_map[mask]
                avg_depth = np.mean(depth_region) if depth_region.size > 0 else 0
                
                # Overlay mask on the output image
                output_image[mask] = cv2.addWeighted(output_image[mask], 0.5, 
                                                    np.full_like(output_image[mask], (0, 255, 0)), 0.5, 0)
                # Add depth text (approximate center of mass)
                y, x = np.where(mask)
                if len(x) > 0 and len(y) > 0:
                    cx, cy = int(np.mean(x)), int(np.mean(y))
                    cv2.putText(output_image, f"Depth: {avg_depth:.2f}", 
                               (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return output_image, depth_map_colored, depth_map

def depth_to_point_cloud(depth_map, intrinsic_matrix):
    """
    Convert depth map to point cloud using intrinsic matrix.
    """
    h, w = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # Create grid of coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack coordinates and flatten
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    
    return points

# Initialize webcam (0 is default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set resolution (optional, adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Intrinsic matrix (approximate values, adjust based on your camera)
intrinsic_matrix = np.array([[500, 0, 320],
                             [0, 500, 240],
                             [0, 0, 1]])

# Choose model type: "detection" or "segmentation"
MODEL_TYPE = "detection"  # Change to "segmentation" if preferred

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Estimate depth and get output
    output_image, depth_map_colored, depth_map = estimate_depth(frame, model_type=MODEL_TYPE)
    
    # Convert depth map to point cloud
    points = depth_to_point_cloud(depth_map, intrinsic_matrix)
    
    # Update Open3D point cloud
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if not added:
        vis.add_geometry(pcd)
        added = True
    else:
        vis.update_geometry(pcd)
    
    # Visualize point cloud
    vis.poll_events()
    vis.update_renderer()
    
    # Display the results
    cv2.imshow("YOLO + Depth Output", output_image)
    cv2.imshow("Depth Map (JET)", depth_map_colored)  # Updated window name
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()