import cv2
import numpy as np
import torch
import open3d as o3d
from ultralytics import YOLO
import time

# Paths to YOLO11 model (segmentation-focused for humans)
SEGMENTATION_MODEL_PATH = "yolo11n-seg.pt"  # Lightweight model for speed

# Load YOLO11 segmentation model
segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)

# Load MiDaS Small model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Initialize Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Real-Time Human Mesh Reconstruction", width=800, height=600)
geom_added = False

def estimate_depth(image):
    """
    Estimate depth and isolate human regions using YOLO11 segmentation.
    Returns human-specific depth map only if a person is detected.
    """
    # Downsample image for speed
    img_resized = cv2.resize(image, (320, 240))
    results = segmentation_model(img_resized, verbose=False)[0]
    
    # Convert to RGB for MiDaS
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    # Depth prediction
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_resized.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Scale depth to metric units (heuristic: assume max depth = 5 meters)
    depth_map_scaled = depth_map / np.max(depth_map) * 5.0
    
    # Process human segmentations
    if results.masks is not None:
        for i, mask in enumerate(results.masks.data):
            if int(results.boxes.cls[i]) == 0:  # Human class (COCO ID 0)
                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (img_resized.shape[1], img_resized.shape[0]))
                mask = mask > 0.5
                human_depth_map = depth_map_scaled * mask
                # Resize back to original resolution
                human_depth_map = cv2.resize(human_depth_map, (image.shape[1], image.shape[0]))
                return human_depth_map
    return None  # No human detected

def depth_to_point_cloud(depth_map, rgb_image, intrinsic_matrix, max_depth=5.0, min_depth=0.1):
    """
    Convert human depth map to a point cloud.
    """
    h, w = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # Create coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack and flatten points
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
    
    # Filter valid points
    z_flat = z.reshape(-1)
    valid = (z_flat > min_depth) & (z_flat < max_depth) & ~np.isnan(z_flat) & ~np.isinf(z_flat)
    points = points[valid]
    colors = colors[valid]
    
    return points, colors

def reconstruct_surface(pcd):
    """
    Create a mesh from the point cloud using Poisson reconstruction.
    """
    if len(pcd.points) < 100:  # Minimum points for reconstruction
        return None
    pcd.estimate_normals()
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    return mesh

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Intrinsic matrix (replace with calibrated values)
intrinsic_matrix = np.array([[500, 0, 320],
                             [0, 500, 240],
                             [0, 0, 1]], dtype=np.float32)

# Point cloud object
pcd = o3d.geometry.PointCloud()

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Estimate depth for human
    human_depth_map = estimate_depth(frame)
    
    if human_depth_map is not None:
        # Convert to point cloud
        points, colors = depth_to_point_cloud(human_depth_map, frame, intrinsic_matrix)
        
        if len(points) > 0:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Reconstruct mesh
            mesh = reconstruct_surface(pcd)
            if mesh:
                if geom_added:
                    vis.remove_geometry(pcd if not mesh else mesh, reset_bounding_box=False)
                vis.add_geometry(mesh)
                geom_added = True
            
            vis.poll_events()
            vis.update_renderer()
    
    # Display original frame with FPS
    cv2.putText(frame, f"FPS: {1 / (time.time() - start_time):.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera Feed", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()

# Optional: Save the final mesh
if geom_added and mesh:
    o3d.io.write_triangle_mesh("human_mesh.ply", mesh)
    print("Saved mesh to 'human_mesh.ply'")
    