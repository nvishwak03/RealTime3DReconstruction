import cv2
import numpy as np
import torch
import open3d as o3d
import os
import time
import argparse
from ultralytics import YOLO

# Set up argument parser
parser = argparse.ArgumentParser(description="Process video for depth estimation and point cloud generation.")
parser.add_argument("--input", type=str, required=True, help="Path to the input video file")
parser.add_argument("--save_pngs", action="store_true", help="Save individual depth frames as PNG files (default: False)")
args = parser.parse_args()

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

# Output directories
output_dir = "output_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
depth_video_path = os.path.join(output_dir, "depth_output.avi")
point_cloud_dir = os.path.join(output_dir, "point_clouds")

if not os.path.exists(point_cloud_dir):
    os.makedirs(point_cloud_dir)

# Initialize Open3D visualizer (optional for real-time, but we can disable for batch processing)
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

def depth_to_point_cloud(depth_map, rgb_image, intrinsic_matrix):
    """
    Convert the entire depth map to a dense point cloud with RGB colors.
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
    
    # Filter invalid points (adjust thresholds for your scene)
    valid = (z.reshape(-1) > 0.01) & (z.reshape(-1) < 500) & ~np.isnan(z.reshape(-1)) & ~np.isinf(z.reshape(-1))
    points = points[valid]
    colors = colors[valid]
    
    return points, colors

# Initialize video capture from file (using command-line argument)
cap = cv2.VideoCapture(args.input)

if not cap.isOpened():
    print(f"Error: Could not open video file: {args.input}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer for depth output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(depth_video_path, fourcc, fps, (frame_width, frame_height))

# Intrinsic matrix (approximate, calibrate for better accuracy)
intrinsic_matrix = np.array([[500, 0, frame_width // 2],
                             [0, 500, frame_height // 2],
                             [0, 0, 1]])

# Choose model type
MODEL_TYPE = "detection"  # or "segmentation"

frame_count = 0
all_points = []  # To accumulate points for a single combined point cloud (optional)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video.")
        break
    
    start_time = time.time()
    
    # Estimate depth and get outputs
    output_image, depth_map_colored, depth_map = estimate_depth(frame, model_type=MODEL_TYPE)
    
    # Convert depth map to point cloud with colors for the entire frame
    points, colors = depth_to_point_cloud(depth_map, frame, intrinsic_matrix)
    
    # Save point cloud for this frame as .ply file
    pcd_frame = o3d.geometry.PointCloud()
    pcd_frame.points = o3d.utility.Vector3dVector(points)
    pcd_frame.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(point_cloud_dir, f"point_cloud_frame_{frame_count:06d}.ply"), pcd_frame)
    
    # Optionally accumulate points for a single combined point cloud
    all_points.append(np.hstack((points, colors)))
    
    # Update Open3D point cloud (optional for real-time visualization during processing)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if not geom_added:
        vis.add_geometry(pcd)
        geom_added = True
    else:
        vis.update_geometry(pcd)
    
    # Render the point cloud (optional, can disable for faster batch processing)
    vis.poll_events()
    vis.update_renderer()
    
    # Save depth estimation output
    out.write(depth_map_colored)  # Save colored depth map as video
    if args.save_pngs:
        cv2.imwrite(os.path.join(output_dir, f"depth_frame_{frame_count:06d}.png"), depth_map_colored)  # Optional: save individual frames
    
    # Display 2D outputs (optional, can disable for faster processing)
    cv2.imshow("YOLO + Depth Output", output_image)
    cv2.imshow("Depth Map", depth_map_colored)
    
    # Print FPS for performance monitoring
    print(f"FPS: {1 / (time.time() - start_time):.2f}")
    
    # Exit on 'q' or continue until video ends
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processing interrupted by user.")
        break
    
    frame_count += 1

# Save combined point cloud (optional)
if all_points:
    combined_points = np.vstack(all_points)[:, :3]  # Only points, not colors for simplicity
    combined_colors = np.vstack(all_points)[:, 3:]  # Colors
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    o3d.io.write_point_cloud(os.path.join(output_dir, "combined_point_cloud.ply"), combined_pcd)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
vis.destroy_window()

print(f"Processed {frame_count} frames. Results saved in {output_dir}")