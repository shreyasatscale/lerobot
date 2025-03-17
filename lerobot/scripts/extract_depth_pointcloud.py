import cv2
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import copy
import math

# Paths to the camera video files - updating to use the static high and low cameras
high_camera_video_path = '/Users/shreyas.chakravarthula/scale/lerobot/recordings/videos/chunk-000/observation.images.cam_high/episode_000000.mp4'
low_camera_video_path = '/Users/shreyas.chakravarthula/scale/lerobot/recordings/videos/chunk-000/observation.images.cam_low/episode_000000.mp4'

# Keeping these for reference
left_wrist_video_path = '/Users/shreyas.chakravarthula/scale/lerobot/recordings/videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4'
right_wrist_video_path = '/Users/shreyas.chakravarthula/scale/lerobot/recordings/videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4'

# Camera intrinsic parameters for Intel RealSense cameras
# D405 for wrist cameras (typical values, these would ideally come from calibration)
d405_fx = 386.0  # D405 focal length x
d405_fy = 386.0  # D405 focal length y
d405_cx = 320.0  # D405 principal point x
d405_cy = 240.0  # D405 principal point y
d405_depth_scale = 0.001  # Convert raw depth values to meters (typical for RealSense)

# D435 for high and low cameras
d435_fx = 616.0  # D435 focal length x
d435_fy = 616.0  # D435 focal length y
d435_cx = 320.0  # D435 principal point x
d435_cy = 240.0  # D435 principal point y
d435_depth_scale = 0.001  # Convert raw depth values to meters

# Initial guess for the transformation between high and low cameras
# Based on physical setup: high camera is ~1m above and ~1m forward of low camera, pointing downward
# This represents a translation of [0, 1.0, 1.0] and rotation of ~90 degrees around X-axis
def create_initial_transform():
    """
    Create an initial transformation matrix based on the known camera setup.
    High camera is ~1m above and ~1m forward from the low camera's perspective,
    and the high camera is pointing downward.
    """
    # Create rotation matrix for ~90 degree rotation around X-axis (pointing down)
    angle_x = math.radians(90)  # Convert to radians
    cos_x = math.cos(angle_x)
    sin_x = math.sin(angle_x)
    
    # Create a 4x4 transformation matrix
    transform = np.eye(4)
    
    # Set rotation part - rotate around X axis to point downward
    transform[1, 1] = cos_x
    transform[1, 2] = -sin_x
    transform[2, 1] = sin_x
    transform[2, 2] = cos_x
    
    # Set translation part - move up and forward
    transform[0, 3] = 0.0    # No horizontal offset (centered)
    transform[1, 3] = 1.0    # 1 meter up
    transform[2, 3] = 1.0    # 1 meter forward
    
    return transform

def extract_frame(video_path, frame_number=0):
    """Extract a specific frame from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    # Make sure the requested frame exists
    if frame_number >= total_frames:
        print(f"Error: Requested frame {frame_number} exceeds total frames {total_frames}")
        cap.release()
        return None
    
    # Set the video to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return None
    
    return frame

def depth_frame_to_pointcloud(depth_frame, camera_type="d435", fx=None, fy=None, cx=None, cy=None, 
                              colorize_by_depth=True, voxel_size=None, apply_filters=True, 
                              outlier_std_ratio=2.0, depth_scale=None):
    """
    Convert a depth frame to a point cloud
    
    Args:
        depth_frame: RGB frame where depth is encoded in the pixel values
        camera_type: Either "d405" (wrist cameras) or "d435" (high/low cameras)
        fx, fy: Focal lengths (override camera defaults if provided)
        cx, cy: Principal point (override camera defaults if provided)
        colorize_by_depth: Whether to color points by depth
        voxel_size: If not None, downsample the point cloud with this voxel size
        apply_filters: Whether to apply filters to remove noise
        outlier_std_ratio: Standard deviation ratio for statistical outlier removal
        depth_scale: Scale factor to convert depth values to meters
    
    Returns:
        Open3D point cloud
    """
    # Set camera parameters based on the camera type
    if camera_type == "d405":
        _fx = d405_fx if fx is None else fx
        _fy = d405_fy if fy is None else fy
        _cx = d405_cx if cx is None else cx
        _cy = d405_cy if cy is None else cy
        _depth_scale = d405_depth_scale if depth_scale is None else depth_scale
    elif camera_type == "d435":
        _fx = d435_fx if fx is None else fx
        _fy = d435_fy if fy is None else fy
        _cx = d435_cx if cx is None else cx
        _cy = d435_cy if cy is None else cy
        _depth_scale = d435_depth_scale if depth_scale is None else depth_scale
    else:
        print(f"Warning: Unknown camera type '{camera_type}', using default values")
        _fx = 500 if fx is None else fx
        _fy = 500 if fy is None else fy
        _cx = 320 if cx is None else cx
        _cy = 240 if cy is None else cy
        _depth_scale = 0.001 if depth_scale is None else depth_scale
    
    print(f"Using camera parameters: fx={_fx}, fy={_fy}, cx={_cx}, cy={_cy}, depth_scale={_depth_scale}")
    
    # For RealSense depth cameras, depth is typically encoded with:
    # - R and G channels for 16-bit depth (R is high byte, G is low byte)
    # - B channel often unused or used for confidence
    
    # Extract depth using RealSense-style encoding
    # Combining R (high byte) and G (low byte) channels for 16-bit depth
    depth_16bit = depth_frame[:, :, 0].astype(np.uint16) * 256 + depth_frame[:, :, 1].astype(np.uint16)
    
    # Convert depth to meters using the depth scale
    depth = depth_16bit.astype(np.float32) * _depth_scale
    
    # Apply bilateral filter for noise reduction while preserving edges
    if apply_filters:
        # Use bilateral filter to preserve edges while reducing noise
        depth_filtered = cv2.bilateralFilter(depth.astype(np.float32), 5, 0.05, 5)
        depth = depth_filtered
    
    # Create coordinate grid
    height, width = depth.shape
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    
    # Calculate 3D coordinates
    z = depth
    x = (x_grid - _cx) * z / _fx
    y = (y_grid - _cy) * z / _fy
    
    # Stack coordinates and reshape
    xyz = np.stack((x, y, z), axis=-1)
    
    # Create point cloud and filter out invalid points
    # For RealSense, zero or very small depth values are invalid
    # Also limit the maximum depth to exclude far points which are usually less accurate
    # D435 has longer range than D405, so adjust the depth range accordingly
    if camera_type == "d435":
        valid_mask = (z > 0.1) & (z < 10.0)  # D435 has longer range (up to ~10m)
    else:
        valid_mask = (z > 0.1) & (z < 3.0)   # D405 works best within 3m
    
    xyz_valid = xyz[valid_mask]
    
    # Handle empty point cloud
    if xyz_valid.shape[0] == 0:
        print("Warning: No valid points found in depth map")
        empty_pcd = o3d.geometry.PointCloud()
        return empty_pcd
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_valid)
    
    # Print depth statistics before filtering
    z_values = xyz_valid[:, 2]
    print(f"Depth statistics before filtering: min={z_values.min():.3f}m, max={z_values.max():.3f}m, mean={z_values.mean():.3f}m")
    
    # Add colors based on depth
    if colorize_by_depth:
        # Normalize z values to 0-1 range for coloring
        z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min())
        
        # Apply a color map (jet: blue=near, red=far)
        colormap = plt.cm.get_cmap('jet')
        colors = colormap(z_norm)[:, :3]  # Get RGB from colormap (drop alpha)
    else:
        # Use a solid color (red) if not coloring by depth
        colors = np.zeros_like(xyz_valid)
        colors[:, 0] = 1.0  # R
        colors[:, 1] = 0.0  # G
        colors[:, 2] = 0.0  # B
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Apply statistical outlier removal to clean up the point cloud
    if apply_filters and len(pcd.points) > 100:
        print("Applying statistical outlier removal...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                std_ratio=outlier_std_ratio)
    
    # Apply voxel downsampling if requested
    if voxel_size is not None and voxel_size > 0 and len(pcd.points) > 100:
        print(f"Downsampling with voxel size {voxel_size}m...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Print statistics after filtering
    if len(pcd.points) > 0:
        points_array = np.asarray(pcd.points)
        z_values_filtered = points_array[:, 2]
        print(f"Depth statistics after filtering: min={z_values_filtered.min():.3f}m, max={z_values_filtered.max():.3f}m, mean={z_values_filtered.mean():.3f}m")
        print(f"Point cloud has {len(pcd.points)} points after filtering")
    
    return pcd

def visualize_pointcloud(pcd, window_name="Point Cloud"):
    """Visualize a point cloud using Open3D"""
    if len(pcd.points) == 0:
        print(f"Warning: Point cloud is empty, cannot visualize")
        return
        
    print(f"Visualizing point cloud with {len(pcd.points)} points")
    
    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # Visualize point cloud with coordinate frame
    o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

def icp_registration(source_pcd, target_pcd, threshold=0.05, max_iteration=100, mode="point_to_point", initial_transform=None):
    """
    Perform ICP registration between two point clouds
    
    Args:
        source_pcd: Source point cloud (will be transformed to align with target)
        target_pcd: Target point cloud
        threshold: Distance threshold for ICP
        max_iteration: Maximum number of iterations
        mode: ICP mode, either "point_to_point" or "point_to_plane"
        initial_transform: Initial transformation guess (4x4 matrix)
    
    Returns:
        Transformation matrix, fitness, RMSE
    """
    if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
        print("Error: Cannot perform ICP on empty point clouds")
        return np.eye(4), 0, float('inf')
        
    print(f"Performing ICP registration with threshold={threshold}, max_iteration={max_iteration}")
    
    # Use provided initial transform or identity if none
    if initial_transform is None:
        initial_transform = np.eye(4)
        
    # Estimate normals if using point-to-plane ICP
    if mode == "point_to_plane":
        print("Estimating normals for point-to-plane ICP...")
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Perform the registration
    if mode == "point_to_plane":
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    else:  # Default to point-to-point
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    
    # Print results
    print(f"ICP Fitness: {result.fitness:.6f}")
    print(f"ICP RMSE: {result.inlier_rmse:.6f}")
    print(f"Transformation matrix:\n{result.transformation}")
    
    return result.transformation, result.fitness, result.inlier_rmse

def main():
    # Extract frames from videos - now using high and low cameras
    print("Extracting frames from videos...")
    high_frame = extract_frame(high_camera_video_path, frame_number=0)
    low_frame = extract_frame(low_camera_video_path, frame_number=0)
    
    if high_frame is None or low_frame is None:
        print("Error: Failed to extract frames")
        return
    
    # Display the frames (for debugging)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("High Camera (D435)")
    plt.imshow(cv2.cvtColor(high_frame, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title("Low Camera (D435)")
    plt.imshow(cv2.cvtColor(low_frame, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.savefig('camera_frames.png')
    plt.close()
    print("Saved camera frames to camera_frames.png")
    
    # Create output directory
    output_dir = Path("/Users/shreyas.chakravarthula/scale/lerobot/outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Convert frames to point clouds with D435 parameters (static cameras)
    print("\nGenerating point cloud for high camera (D435)...")
    high_pcd = depth_frame_to_pointcloud(
        high_frame, 
        camera_type="d435",
        colorize_by_depth=True,
        voxel_size=0.01,  # 1cm voxels for D435 (which is reasonable for environment-scale scenes)
        apply_filters=True
    )
    
    print("\nGenerating point cloud for low camera (D435)...")
    low_pcd = depth_frame_to_pointcloud(
        low_frame, 
        camera_type="d435",
        colorize_by_depth=True,
        voxel_size=0.01,  # 1cm voxels
        apply_filters=True
    )
    
    # Save the original point clouds
    o3d.io.write_point_cloud(str(output_dir / "high_camera_pointcloud.ply"), high_pcd)
    o3d.io.write_point_cloud(str(output_dir / "low_camera_pointcloud.ply"), low_pcd)
    print(f"Original point clouds saved to {output_dir}")
    
    # Create a depth color legend image and save it
    plt.figure(figsize=(6, 1))
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    plt.imshow(gradient, aspect='auto', cmap='jet')
    plt.title('Depth Color Map: Blue (Near) to Red (Far)')
    plt.tick_params(left=False, right=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'depth_colormap.png'))
    plt.close()
    print("Saved depth colormap legend to depth_colormap.png")
    
    # Visualize original point clouds
    print("\nVisualizing high camera point cloud...")
    visualize_pointcloud(high_pcd, "High Camera Point Cloud (D435, Colored by Depth)")
    print("\nVisualizing low camera point cloud...")
    visualize_pointcloud(low_pcd, "Low Camera Point Cloud (D435, Colored by Depth)")
    
    # Perform ICP registration between the static cameras
    print("\nPerforming ICP registration between high and low camera point clouds...")
    
    # Create initial transform based on known camera positions
    initial_transform = create_initial_transform()
    print("\nUsing initial transformation guess based on camera setup:")
    print(initial_transform)
    
    # Multi-stage ICP for better alignment
    # First, use a larger threshold to get rough alignment
    print("\n--- Stage 1: Rough alignment with point-to-point ICP ---")
    transform_rough, fitness_rough, rmse_rough = icp_registration(
        high_pcd, low_pcd, threshold=1.5, max_iteration=100, mode="point_to_point", 
        initial_transform=initial_transform
    )
    
    # Apply the rough transformation to the source for better starting point
    high_pcd_rough = copy.deepcopy(high_pcd)
    high_pcd_rough.transform(transform_rough)
    
    # Then refine with point-to-plane with tighter threshold
    print("\n--- Stage 2: Fine alignment with point-to-plane ICP ---")
    transform_fine, fitness_fine, rmse_fine = icp_registration(
        high_pcd_rough, low_pcd, threshold=0.20, max_iteration=100, mode="point_to_plane"
    )
    
    # Combine the transformations
    transform = np.matmul(transform_fine, transform_rough)
    
    print("\nFinal transformation (combination of rough and fine):")
    print(transform)
    
    # Transform the high camera point cloud using the combined transformation
    high_pcd_transformed = copy.deepcopy(high_pcd)
    high_pcd_transformed.transform(transform)
    
    # Save the transformed point cloud
    o3d.io.write_point_cloud(
        str(output_dir / "high_camera_transformed.ply"), 
        high_pcd_transformed
    )
    
    # Save the transformation matrix to a file
    np.savetxt(
        str(output_dir / "static_cameras_transformation.txt"),
        transform,
        header="# Transformation matrix from high D435 camera to low D435 camera",
        fmt='%.8f'
    )
    print(f"Transformation matrix saved to {output_dir}/static_cameras_transformation.txt")
    
    # Visualize the aligned point clouds
    # Create a combined point cloud with different colors for high and low
    high_pcd_transformed.paint_uniform_color([1, 0, 0])  # Red for high camera (transformed)
    low_pcd.paint_uniform_color([0, 0, 1])              # Blue for low camera
    
    combined_pcd = high_pcd_transformed + low_pcd
    
    print("\nVisualizing aligned point clouds (red: high transformed, blue: low)...")
    visualize_pointcloud(combined_pcd, "Aligned Point Clouds (After ICP)")
    
    # Save the combined point cloud
    o3d.io.write_point_cloud(
        str(output_dir / "combined_static_cameras_pointcloud.ply"),
        combined_pcd
    )
    print(f"Combined aligned point cloud saved to {output_dir}/combined_static_cameras_pointcloud.ply")
    
    print("\nStatic cameras calibration complete!")

if __name__ == "__main__":
    main() 