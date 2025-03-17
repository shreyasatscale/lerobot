from lerobot.common.robot_devices.cameras.intelrealsense import find_cameras, save_images_from_cameras
from pathlib import Path

cameras = find_cameras()

for camera in cameras:
    print(f"Name: {camera['name']}, Serial Number: {camera['serial_number']}")
    
output_dir = Path("camera_output")

save_images_from_cameras(
    images_dir=output_dir,
    fps=30,
    width=640,
    height=480,
    record_time_s=3
)
