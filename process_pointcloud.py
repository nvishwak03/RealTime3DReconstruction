import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- Configuration ----------------------
PLY_INPUT_PATH = "output_point_cloud.ply"
PLY_OUTPUT_PATH = "enhanced_output.ply"
ENABLE_HIDDEN_REMOVAL = True
ENABLE_PATCH_DETECTION = True
ENABLE_CLUSTERING = True
VOXEL_SIZE = 0.02
# -----------------------------------------------------------

# Load PLY point cloud
pcd = o3d.io.read_point_cloud(PLY_INPUT_PATH)
print(f"Original point cloud has {len(pcd.points)} points.")

# Downsample
downpcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

# Estimate normals
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Hidden point removal (optional)
if ENABLE_HIDDEN_REMOVAL:
    diameter = np.linalg.norm(downpcd.get_max_bound() - downpcd.get_min_bound())
    camera = [0, 0, diameter]
    radius = diameter * 100
    _, pt_map = downpcd.hidden_point_removal(camera, radius)
    downpcd = downpcd.select_by_index(pt_map)
    print(f"After hidden point removal: {len(downpcd.points)} points")

# Planar patch detection (optional)
geometries = [downpcd]
if ENABLE_PATCH_DETECTION:
    print("Detecting planar patches...")
    patches = downpcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0,
        min_num_points=0,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )
    for obox in patches:
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(obox.color)
        geometries.append(mesh)
        geometries.append(obox)
    print(f"Detected {len(patches)} planar patches.")

# DBSCAN clustering (optional)
if ENABLE_CLUSTERING:
    print("Running DBSCAN clustering...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        labels = np.array(downpcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    max_label = labels.max()
    print(f"Found {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # noise
    downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Save enhanced point cloud
o3d.io.write_point_cloud(PLY_OUTPUT_PATH, downpcd)
print(f"Saved enhanced point cloud to {PLY_OUTPUT_PATH}")

# Visualize everything
o3d.visualization.draw_geometries(geometries,
    point_show_normal=True,
    zoom=0.5,
    front=[0.5, 0, -1],
    lookat=downpcd.get_center().tolist(),
    up=[0, -1, 0]
)
