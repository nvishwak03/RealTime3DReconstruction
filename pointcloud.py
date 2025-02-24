import cv2
import numpy as np
import torch
import open3d as o3d
from ultralytics import YOLO
import time

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
vis.create_window(window_name="Real-Time Point Cloud", width=800, height=600)
pcd = o3d.geometry.PointCloud()
geom_added = False

def estimate_depth(image, model_type="detection"):
    """
    Estimate depth for the entire image and optionally overlay YOLO results.
    """
    model = detection_model if model_type == "detection" else segmentation_model
    results = model(image, verbose=False)[0]
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)
    
    output_image = image.copy()
    
    if model_type == "detection":
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{results.names[cls]} {conf:.2f}"
            depth_region = depth_map[y1:y2, x1:x2]
            avg_depth = np.mean(depth_region) if depth_region.size > 0 else 0
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_image, f"{label} Depth: {avg_depth:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    elif model_type == "segmentation":
        if results.masks is not None:
            for mask in results.masks.data:
                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask = mask > 0.5
                depth_region = depth_map[mask]
                avg_depth = np.mean(depth_region) if depth_region.size > 0 else 0
                output_image[mask] = cv2.addWeighted(output_image[mask], 0.5, 
                                                    np.full_like(output_image[mask], (0, 255, 0)), 0.5, 0)
                y, x = np.where(mask)
                if len(x) > 0 and len(y) > 0:
                    cx, cy = int(np.mean(x)), int(np.mean(y))
                    cv2.putText(output_image, f"Depth: {avg_depth:.2f}", 
                                (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return output_image, depth_map_colored, depth_map

def depth_to_point_cloud(depth_map, rgb_image, intrinsic_matrix, max_depth=1000.0, min_depth=0.0):
    """
    Convert the depth map to a point cloud, with relaxed filtering for debugging.
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
    
    # Get RGB colors from the original image (normalized to [0, 1])
    colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
    
    # Debug: Print raw depth range
    z_flat = z.reshape(-1)
    print(f"Raw depth range: {np.min(z_flat):.2f} to {np.max(z_flat):.2f}")
    
    # Filter points based on depth thresholds (relaxed for now)
    valid = (z_flat > min_depth) & (z_flat < max_depth) & ~np.isnan(z_flat) & ~np.isinf(z_flat)
    points = points[valid]
    colors = colors[valid]
    
    # Debug: Print point cloud stats
    print(f"Number of valid points: {len(points)}")
    if len(points) > 0:
        print(f"Filtered depth range: {z_flat[valid].min():.2f} to {z_flat[valid].max():.2f}")
    
    return points, colors

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Intrinsic matrix (approximate, calibrate for better accuracy)
intrinsic_matrix = np.array([[500, 0, 320],
                             [0, 500, 240],
                             [0, 0, 1]])

# Choose model type
MODEL_TYPE = "detection"  # or "segmentation"

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Estimate depth and get outputs
    output_image, depth_map_colored, depth_map = estimate_depth(frame, model_type=MODEL_TYPE)
    
    # Convert depth map to point cloud with relaxed filtering
    points, colors = depth_to_point_cloud(depth_map, frame, intrinsic_matrix, max_depth=1000.0, min_depth=0.0)
    
    # Check if points exist before updating
    if len(points) > 0:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if not geom_added:
            vis.add_geometry(pcd)
            geom_added = True
        else:
            vis.update_geometry(pcd)
        
        # Render the point cloud
        vis.poll_events()
        vis.update_renderer()
    else:
        print("Warning: No valid points for point cloud.")
    
    # Display 2D outputs
    cv2.imshow("YOLO + Depth Output", output_image)
    cv2.imshow("Depth Map", depth_map_colored)
    
    # Print FPS for performance monitoring
    print(f"FPS: {1 / (time.time() - start_time):.2f}")
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()