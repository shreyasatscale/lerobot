import pyrealsense2 as rs
import numpy as np
import os
import json
from datetime import datetime
import cv2


def create_directory_structure(base_path, camera_name):
    """Create directory structure for a camera"""
    camera_path = os.path.join(base_path, camera_name)
    os.makedirs(os.path.join(camera_path, "pointclouds"), exist_ok=True)
    os.makedirs(os.path.join(camera_path, "depth"), exist_ok=True)
    os.makedirs(os.path.join(camera_path, "rgb"), exist_ok=True)
    return camera_path


def save_intrinsics(camera_path, color_profile, depth_profile, device):
    """Save camera intrinsics and additional camera information to JSON file"""
    # Try to get calibration data
    try:
        # Get advanced mode
        advanced_mode = rs.rs400_advanced_mode(device)
        if advanced_mode.is_enabled():
            print("Advanced mode is enabled")
            # Try to get calibration data using different methods
            try:
                # Get calibration data from advanced mode
                calib_data = advanced_mode.get_calibration_data()
                print("\n=== Calibration Data ===")
                print(f"Calibration data: {calib_data}")
               
                # Try to get specific calibration parameters
                try:
                    color_calib = advanced_mode.get_calibration_data(rs.stream.color)
                    depth_calib = advanced_mode.get_calibration_data(rs.stream.depth)
                    print(f"\nColor Calibration: {color_calib}")
                    print(f"Depth Calibration: {depth_calib}")
                except Exception as e:
                    print(f"Could not get stream-specific calibration: {e}")
                   
                # Try to get calibration table as string
                try:
                    calib_table_str = advanced_mode.get_calibration_table_string()
                    print(f"\nCalibration Table String: {calib_table_str}")
                except Exception as e:
                    print(f"Could not get calibration table string: {e}")
                   
            except Exception as e:
                print(f"Could not get calibration data: {e}")
        else:
            print("Advanced mode is not enabled")
            # Try to enable advanced mode
            try:
                advanced_mode.toggle_advanced_mode(True)
                print("Enabled advanced mode")
            except Exception as e:
                print(f"Could not enable advanced mode: {e}")
    except Exception as e:
        print(f"Could not access advanced mode: {e}")


    color_intrinsics = color_profile.get_intrinsics()
    depth_intrinsics = depth_profile.get_intrinsics()
   
    # Debug printing for intrinsics
    print("\n=== Color Camera Intrinsics ===")
    print(f"Width: {color_intrinsics.width}")
    print(f"Height: {color_intrinsics.height}")
    print(f"PPX: {color_intrinsics.ppx}")
    print(f"PPY: {color_intrinsics.ppy}")
    print(f"FX: {color_intrinsics.fx}")
    print(f"FY: {color_intrinsics.fy}")
    print(f"Model: {color_intrinsics.model}")
    print(f"Distortion Coefficients: {color_intrinsics.coeffs}")
    print(f"Number of distortion coefficients: {len(color_intrinsics.coeffs)}")
   
    print("\n=== Depth Camera Intrinsics ===")
    print(f"Width: {depth_intrinsics.width}")
    print(f"Height: {depth_intrinsics.height}")
    print(f"PPX: {depth_intrinsics.ppx}")
    print(f"PPY: {depth_intrinsics.ppy}")
    print(f"FX: {depth_intrinsics.fx}")
    print(f"FY: {depth_intrinsics.fy}")
    print(f"Model: {depth_intrinsics.model}")
    print(f"Distortion Coefficients: {depth_intrinsics.coeffs}")
    print(f"Number of distortion coefficients: {len(depth_intrinsics.coeffs)}")
   
    # Get additional device information
    device_info = {
        "name": device.get_info(rs.camera_info.name),
        "serial_number": device.get_info(rs.camera_info.serial_number),
        "firmware_version": device.get_info(rs.camera_info.firmware_version),
        "usb_type": device.get_info(rs.camera_info.usb_type_descriptor),
        "product_id": device.get_info(rs.camera_info.product_id),
        "product_line": device.get_info(rs.camera_info.product_line)
    }
   
    # Get stream profiles
    color_stream_profile = color_profile.as_video_stream_profile()
    depth_stream_profile = depth_profile.as_video_stream_profile()
   
    intrinsics_data = {
        "device_info": device_info,
        "color_camera": {
            "width": int(color_intrinsics.width),
            "height": int(color_intrinsics.height),
            "ppx": float(color_intrinsics.ppx),
            "ppy": float(color_intrinsics.ppy),
            "fx": float(color_intrinsics.fx),
            "fy": float(color_intrinsics.fy),
            "model": str(color_intrinsics.model),
            "coeffs": [float(x) for x in color_intrinsics.coeffs],
            "fps": color_stream_profile.fps(),
            "format": str(color_stream_profile.format()),
            "index": color_stream_profile.stream_index(),
            "unique_id": color_stream_profile.unique_id(),
            "stream_type": str(color_stream_profile.stream_type()),
            "is_default": color_stream_profile.is_default()
        },
        "depth_camera": {
            "width": int(depth_intrinsics.width),
            "height": int(depth_intrinsics.height),
            "ppx": float(depth_intrinsics.ppx),
            "ppy": float(depth_intrinsics.ppy),
            "fx": float(depth_intrinsics.fx),
            "fy": float(depth_intrinsics.fy),
            "model": str(depth_intrinsics.model),
            "coeffs": [float(x) for x in depth_intrinsics.coeffs],
            "fps": depth_stream_profile.fps(),
            "format": str(depth_stream_profile.format()),
            "index": depth_stream_profile.stream_index(),
            "unique_id": depth_stream_profile.unique_id(),
            "stream_type": str(depth_stream_profile.stream_type()),
            "is_default": depth_stream_profile.is_default()
        },
        "extrinsics_depth_to_color": {
            "rotation": [float(x) for x in depth_profile.get_extrinsics_to(color_profile).rotation],
            "translation": [float(x) for x in depth_profile.get_extrinsics_to(color_profile).translation]
        }
    }
   
    with open(os.path.join(camera_path, "calibration.json"), 'w') as f:
        json.dump(intrinsics_data, f, indent=4)


# Create timestamped base directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_path = os.path.join("recordings", timestamp)
os.makedirs(base_path, exist_ok=True)


# Get list of connected devices
ctx = rs.context()
devices = ctx.query_devices()
print(f"Found {len(devices)} RealSense devices")


# Process each camera
for device_index, device in enumerate(devices):
    device_name = f"camera_{device_index}"
    device_serial = device.get_info(rs.camera_info.serial_number)
    print(f"\nProcessing {device_name} (Serial: {device_serial})")
   
    # Create directory structure for this camera
    camera_path = create_directory_structure(base_path, device_name)
   
    # Configure pipeline for this device
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device_serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
   
    # Start streaming
    profile = pipeline.start(config)
   
    # Get profiles and save intrinsics
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    save_intrinsics(camera_path, color_profile, depth_profile, device)
   
    # Create align object
    align_to = rs.stream.color
    align = rs.align(align_to)
   
    # Counter for number of captures
    capture_count = 0
    num_captures = 5  # Number of captures per camera
   
    try:
        while True:
            # Get aligned frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
           
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
           
            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
           
            # Generate point cloud
            pc = rs.pointcloud()
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
           
            # Save point cloud with proper extension
            ply_path = os.path.join(camera_path, "pointclouds", f"pointcloud_{capture_count:02d}.ply")
            ply = rs.save_to_ply(ply_path)
            ply.set_option(rs.save_to_ply.option_ply_binary, True)
            ply.set_option(rs.save_to_ply.option_ply_normals, False)
            ply.process(points)
           
            # Save depth image with proper extension
            depth_path = os.path.join(camera_path, "depth", f"depth_{capture_count:02d}.png")
            cv2.imwrite(depth_path, depth_image)
           
            # Save RGB image with proper extension
            rgb_path = os.path.join(camera_path, "rgb", f"rgb_{capture_count:02d}.png")
            cv2.imwrite(rgb_path, color_image)
           
            print(f"Captured frame {capture_count + 1}/{num_captures} for {device_name}")
            capture_count += 1
           
            # Break after capturing desired number of frames
            if capture_count >= num_captures:
                break
           
    finally:
        pipeline.stop()


print(f"\nRecording complete! Data saved in: {base_path}")





