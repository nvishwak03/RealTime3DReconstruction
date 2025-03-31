import cv2
import numpy as np
import torch
import open3d as o3d
from ultralytics import YOLO
import time
import os

# Paths to your YOLO11 models
DETECTION_MODEL_PATH = "yolo11n.pt"
SEGMENTATION_MODEL_PATH = "yolo11n-seg.pt"

def load_models():
    """Load all required models"""
    detection_model = YOLO(DETECTION_MODEL_PATH)
    segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    return detection_model, segmentation_model, midas, device, transform

def estimate_depth(image, detection_model, segmentation_model, midas, device, transform, model_type="detection"):
    """
    Estimate depth for the entire image and overlay YOLO results.
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
    """Convert depth map to point cloud"""
    h, w = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
    
    z_flat = z.reshape(-1)
    valid = (z_flat > min_depth) & (z_flat < max_depth) & ~np.isnan(z_flat) & ~np.isinf(z_flat)
    points = points[valid]
    colors = colors[valid]
    
    return points, colors

def process_and_save_video(video_path, output_dir, model_type="detection"):
    """Process video and save results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    detection_model, segmentation_model, midas, device, transform = load_models()
    
    # Intrinsic matrix (approximate)
    intrinsic_matrix = np.array([[500, 0, 320],
                                [0, 500, 240],
                                [0, 0, 1]])
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(
        os.path.join(output_dir, 'output_video.avi'),
        fourcc, fps, (width, height)
    )
    depth_video = cv2.VideoWriter(
        os.path.join(output_dir, 'depth_video.avi'),
        fourcc, fps, (width, height)
    )
    
    frame_idx = 0
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break
            
        # Process frame
        output_image, depth_map_colored, depth_map = estimate_depth(
            frame, detection_model, segmentation_model, midas, device, transform, model_type
        )
        
        # Convert to point cloud and save
        points, colors = depth_to_point_cloud(depth_map, frame, intrinsic_matrix)
        if len(points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(
                os.path.join(output_dir, f'point_cloud_{frame_idx:06d}.ply'),
                pcd
            )
        
        # Write frames to video files
        output_video.write(output_image)
        depth_video.write(depth_map_colored)
        
        frame_idx += 1
        print(f"Processed frame {frame_idx}/{frame_count} - FPS: {1 / (time.time() - start_time):.2f}")
    
    # Cleanup
    cap.release()
    output_video.release()
    depth_video.release()
    print(f"Results saved to {output_dir}")
    print(f"Total frames processed: {frame_idx}")

if __name__ == "__main__":
    # Example usage
    video_path = "5834623-uhd_3840_2160_24fps.mp4"  # Replace with your video file path
    output_dir = "output_results"          # Directory where results will be saved
    process_and_save_video(video_path, output_dir, model_type="detection")