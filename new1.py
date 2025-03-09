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
vis.create_window(window_name="Real-Time Human 3D Reconstruction", width=800, height=600)
pcd = o3d.geometry.PointCloud()
mesh = o3d.geometry.TriangleMesh()
geom_added = False

# Depth buffer for temporal smoothing
depth_buffer = []
MAX_BUFFER_SIZE = 5

def estimate_depth(image):
    """
    Estimate depth and isolate human regions using YOLO11 segmentation.
    Returns full depth map and human-specific depth maps.
    """
    # Downsample image for speed (optional, adjust as needed)
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
    
    # Temporal smoothing
    global depth_buffer
    depth_buffer.append(depth_map)
    if len(depth_buffer) > MAX_BUFFER_SIZE:
        depth_buffer.pop(0)
    smoothed_depth = np.mean(depth_buffer, axis=0)
    
    # Normalize and colorize depth map for visualization
    depth_map_normalized = cv2.normalize(smoothed_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)
    
    # Scale depth to metric units (heuristic: assume max depth = 5 meters)
    depth_map_scaled = smoothed_depth / np.max(smoothed_depth) * 5.0
    
    output_image = img_resized.copy()
    human_depth_maps = []
    
    # Process human segmentations
    if results.masks is not None:
        for i, mask in enumerate(results.masks.data):
            if int(results.boxes.cls[i]) == 0:  # Human class (COCO ID 0)
                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (img_resized.shape[1], img_resized.shape[0]))
                mask = mask > 0.5
                depth_region = depth_map_scaled[mask]
                avg_depth = np.mean(depth_region) if depth_region.size > 0 else 0
                human_depth_maps.append(depth_map_scaled * mask)  # Masked depth map
                
                # Overlay segmentation on output image
                output_image[mask] = cv2.addWeighted(output_image[mask], 0.5, 
                                                    np.full_like(output_image[mask], (0, 255, 0)), 0.5, 0)
                y, x = np.where(mask)
                if len(x) > 0 and len(y) > 0:
                    cx, cy = int(np.mean(x)), int(np.mean(y))
                    cv2.putText(output_image, f"Depth: {avg_depth:.2f}m", 
                                (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Resize outputs back to original resolution
    output_image = cv2.resize(output_image, (image.shape[1], image.shape[0]))
    depth_map_colored = cv2.resize(depth_map_colored, (image.shape[1], image.shape[0]))
    human_depth_maps = [cv2.resize(d, (image.shape[1], image.shape[0])) for d in human_depth_maps]
    
    return output_image, depth_map_colored, depth_map_scaled, human_depth_maps

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
    
    print(f"Number of valid points: {len(points)}")
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

# Global point cloud for multi-view fusion (optional)
global_pcd = o3d.geometry.PointCloud()

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Estimate depth and isolate humans
    output_image, depth_map_colored, depth_map_scaled, human_depth_maps = estimate_depth(frame)
    
    # Process human depth maps
    if human_depth_maps:
        for human_depth in human_depth_maps:
            points, colors = depth_to_point_cloud(human_depth, frame, intrinsic_matrix, max_depth=5.0, min_depth=0.1)
            if len(points) > 0:
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
                # Optional: Accumulate for multi-view
                global_pcd += pcd
                global_pcd = global_pcd.voxel_down_sample(voxel_size=0.01)
                
                # Update point cloud visualization
                if not geom_added:
                    vis.add_geometry(pcd)
                    geom_added = True
                else:
                    vis.update_geometry(pcd)
                
                # Generate mesh periodically (e.g., every 10 frames)
                if int(time.time() * 10) % 100 == 0:  # Adjust frequency
                    mesh = reconstruct_surface(pcd)
                    if mesh:
                        vis.remove_geometry(pcd)
                        vis.add_geometry(mesh)
                        geom_added = True
                
                vis.poll_events()
                vis.update_renderer()
    
    # Display 2D outputs
    cv2.imshow("Human Segmentation + Depth", output_image)
    cv2.imshow("Depth Map", depth_map_colored)
    
    # Print FPS
    print(f"FPS: {1 / (time.time() - start_time):.2f}")
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()

# Optional: Save the final point cloud or mesh
if len(global_pcd.points) > 0:
    o3d.io.write_point_cloud("human_reconstruction.ply", global_pcd)
    print("Saved point cloud to 'human_reconstruction.ply'")