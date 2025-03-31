import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO

# --- Configuration ---
IMAGE_PATH = "input.jpg"  # Set your image path here
MODEL_TYPE = "segmentation"  # Choose "detection" or "segmentation"
PLY_SAVE_PATH = "output_point_cloud.ply"

# Intrinsic matrix (replace with calibrated one if available)
intrinsic_matrix = np.array([[500, 0, 320],
                             [0, 500, 240],
                             [0, 0, 1]])

# Load image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Load YOLO model
MODEL_PATH = "yolo11n-seg.pt" if MODEL_TYPE == "segmentation" else "yolo11n.pt"
model = YOLO(MODEL_PATH)

# Run inference
results = model(image, verbose=False)[0]

# Create empty mask
mask_combined = np.zeros(image.shape[:2], dtype=np.uint8)

# Generate combined mask from results
if MODEL_TYPE == "segmentation" and results.masks is not None:
    for mask in results.masks.data:
        m = mask.cpu().numpy()
        m = cv2.resize(m, (image.shape[1], image.shape[0]))
        m = (m > 0.5).astype(np.uint8)
        mask_combined = np.maximum(mask_combined, m)
else:  # detection boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        mask_combined[y1:y2, x1:x2] = 1

# Assign a fake depth (Z=1.0) to all valid pixels
fake_depth = np.ones_like(mask_combined, dtype=np.float32)

# Create meshgrid of coordinates
h, w = fake_depth.shape
u, v = np.meshgrid(np.arange(w), np.arange(h))

# Intrinsic parameters
fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

z = fake_depth
x = (u - cx) * z / fx
y = (v - cy) * z / fy

# Stack into 3D points
points = np.stack((x, y, z), axis=-1)
points = points[mask_combined == 1]

# Get color values
colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
colors = colors[mask_combined == 1]
colors = colors.astype(np.float32) / 255.0

# Create Open3D PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save to PLY
o3d.io.write_point_cloud(PLY_SAVE_PATH, pcd)
print(f"Saved point cloud to {PLY_SAVE_PATH}")
