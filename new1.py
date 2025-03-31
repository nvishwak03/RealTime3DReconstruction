import cv2
import numpy as np
import torch
import open3d as o3d
from ultralytics import YOLO
import time

# Load YOLO11 segmentation model
segmentation_model = YOLO("yolo11n-seg.pt")

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cuda" if torch.cuda.is_available() else "cpu").eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Camera intrinsics (use real calibration if possible)
intrinsic_matrix = np.array([[500, 0, 320],
                             [0, 500, 240],
                             [0, 0, 1]], dtype=np.float32)

# Open3D window
vis = o3d.visualization.Visualizer()
vis.create_window("Human Mesh Reconstruction", width=800, height=600)
mesh_geom = None

# Accumulated point cloud
accumulated_pcd = o3d.geometry.PointCloud()

def smooth_depth_map(depth_map):
    return cv2.bilateralFilter(depth_map.astype(np.float32), d=5, sigmaColor=75, sigmaSpace=75)

def estimate_depth(image):
    img_resized = cv2.resize(image, (320, 240))
    results = segmentation_model(img_resized, verbose=False)[0]

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img_resized.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = np.flipud(depth_map / np.max(depth_map) * 5.0)

    if results.masks is not None:
        for i, mask in enumerate(results.masks.data):
            if int(results.boxes.cls[i]) == 0:
                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (img_resized.shape[1], img_resized.shape[0])) > 0.5
                mask = np.flipud(mask.astype(np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

                human_depth = depth_map * mask
                human_depth = cv2.resize(human_depth, (image.shape[1], image.shape[0]))
                return smooth_depth_map(human_depth), mask
    return None, None

def depth_to_point_cloud(depth_map, rgb_image):
    h, w = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    v = h - 1 - v
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    rgb_image_flipped = np.flipud(rgb_image)
    colors = cv2.cvtColor(rgb_image_flipped, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

    z_flat = z.reshape(-1)
    valid = (z_flat > 0.1) & (z_flat < 5.0)
    return points[valid], colors[valid]

def preprocess_pcd(pcd):
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(0.01)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
    return pcd

def reconstruct_mesh(pcd):
    if len(pcd.points) < 100:
        return None
    pcd = preprocess_pcd(pcd)
    if len(pcd.points) < 100:
        return None
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=5000)
        return mesh
    except:
        return None

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    depth_map, mask = estimate_depth(frame)
    if depth_map is not None:
        points, colors = depth_to_point_cloud(depth_map, frame)
        if len(points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            accumulated_pcd += pcd

    # Reconstruct mesh if enough points
    if len(accumulated_pcd.points) > 10000:
        mesh = reconstruct_mesh(accumulated_pcd)
        if mesh:
            if mesh_geom:
                vis.remove_geometry(mesh_geom, reset_bounding_box=False)
            vis.clear_geometries()
            vis.add_geometry(mesh)
            mesh_geom = mesh
            accumulated_pcd.clear()

    vis.poll_events()
    vis.update_renderer()

    # Show frame with FPS
    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera Feed", frame)

    # Optional: show debug depth/mask
    if depth_map is not None and mask is not None:
        cv2.imshow("Depth Map", cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        cv2.imshow("Mask", (mask * 255).astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()

# Save mesh
if mesh_geom:
    o3d.io.write_triangle_mesh("human_mesh.ply", mesh_geom)
    print("Mesh saved to 'human_mesh.ply'")
