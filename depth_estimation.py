import cv2
import numpy as np
import torch
from ultralytics import YOLO
import open3d as o3d

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

def generate_point_cloud(depth_map, image, fx=640, fy=640, cx=320, cy=240):
    h, w = depth_map.shape
    points = []
    colors = []

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    y, x = np.indices((h, w))

    z = depth_map
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = img_rgb.reshape(-1, 3) / 255.0

    # Filter invalid points (NaN, inf, or near-zero depth)
    valid = (z.reshape(-1) > 0.1) & np.isfinite(z.reshape(-1))
    points = points[valid]
    colors = colors[valid]

    # Downsample for performance
    if len(points) > 100000:
        indices = np.random.choice(len(points), 100000, replace=False)
        points = points[indices]
        colors = colors[indices]

    print(f"Generated {len(points)} points.")  # Debug point count
    return points, colors

def estimate_depth_and_point_cloud(image, model_type="detection"):
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
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)
    
    points, colors = generate_point_cloud(depth_map, image)
    
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
    
    return output_image, depth_map_colored, points, colors

# Initialize Open3D Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window("Real-Time 3D Point Cloud", width=800, height=600)
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
print("Open3D window initialized with MiDaS Small. Look for a window titled 'Real-Time 3D Point Cloud'.")

# Initialize webcam (0 is default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set resolution (optional, adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Choose model type: "detection" or "segmentation"
MODEL_TYPE = "detection"  # Change to "segmentation" if preferred
frame_count = 0
first_run = True

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    frame_count += 1

    output_image, depth_map, points, colors = estimate_depth_and_point_cloud(frame, model_type=MODEL_TYPE)
    
    # Update point cloud every 5 frames
    if frame_count % 5 == 0:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        
        # Adjust view on first update
        if first_run:
            view_control = vis.get_view_control()
            view_control.set_zoom(0.5)
            view_control.set_front([0, 0, -1])
            view_control.set_up([0, 1, 0])
            first_run = False
        
        print(f"Frame {frame_count}: Updated point cloud with {len(points)} points.")
        
        # Save first point cloud for verification
        if frame_count == 5:
            o3d.io.write_point_cloud("first_point_cloud_midas_small.ply", pcd)
            print("Saved first point cloud to 'first_point_cloud_midas_small.ply' for verification.")
    
    # Display 2D results
    cv2.imshow("YOLO + Depth Output", output_image)
    cv2.imshow("Depth Map", depth_map)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
print("Program terminated.")