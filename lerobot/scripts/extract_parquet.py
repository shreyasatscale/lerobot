import pandas as pd
import open3d as o3d
import numpy as np

# Load the parquet file
df = pd.read_parquet('/Users/shreyas.chakravarthula/scale/lerobot/recordings/data/chunk-000/episode_000000.parquet')

# Extract observation.state column which contains numpy arrays
observation_states = df['observation.state'].to_numpy()

# Assuming the first three elements of each observation.state array represent x, y, z coordinates
# Extract these into a new array for the point cloud
points = np.array([state[:3] for state in observation_states])

print(f"Shape of extracted points: {points.shape}")
print(f"Sample points: {points[:3]}")

# Create a point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Assign some colors (random or based on position)
# Using position-based coloring
colors = np.zeros_like(points)
colors[:, 0] = (points[:, 0] - np.min(points[:, 0])) / (np.max(points[:, 0]) - np.min(points[:, 0]))  # R
colors[:, 1] = (points[:, 1] - np.min(points[:, 1])) / (np.max(points[:, 1]) - np.min(points[:, 1]))  # G
colors[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2]))  # B
pcd.colors = o3d.utility.Vector3dVector(colors)

print("Visualizing point cloud...")
o3d.visualization.draw_geometries([pcd])
