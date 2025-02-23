import open3d as o3d
pcd = o3d.io.read_point_cloud("point_cloud_10.ply")
print(f"Loaded {len(pcd.points)} points from PLY.")
o3d.visualization.draw_geometries([pcd])