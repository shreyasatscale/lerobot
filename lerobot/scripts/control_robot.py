"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.
If the dataset you want to extend is not on the hub, you also need to add `--control.local_files_only=true`.

- Train on this dataset with the ACT policy:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

import logging
import time
import os
import shutil
from dataclasses import asdict, dataclass
from pprint import pformat
from pathlib import Path
from datetime import datetime
import pandas as pd
import cv2
import numpy as np

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

@dataclass
class RecordDepthControlConfig:
    """Configuration for recording depth and RGB images."""
    root: str = "recordings"  # Root directory for saving recordings
    num_frames: int = 3  # Number of frames to capture
    delay_between_frames: float = 1.0  # Delay between frames in seconds

def reorganize_videos(root_dir, recorded_episodes):
    """
    Reorganize videos into the structure:
    sequence-timestamp/
      episode_0/
        cam_high.mp4
        cam_left_wrist.mp4
        ...
      episode_1/
        ...
    """
    try:
        root_path = Path(root_dir)
        
        # Create a unique sequence name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sequence_name = f"sequence-{timestamp}"
        
        # Create a proper directory for reorganized files
        sequence_dir_path = root_path.parent / sequence_name
        os.makedirs(sequence_dir_path, exist_ok=True)
        print(f"Creating reorganized videos in {sequence_dir_path}")
        
        # Get the videos directory path
        videos_dir = root_path / "videos"
        
        if not videos_dir.exists():
            print(f"Videos directory not found at {videos_dir}")
            return None
            
        # Find the first chunk directory (typically chunk-000)
        chunk_dirs = [d for d in videos_dir.iterdir() if d.is_dir()]
        if not chunk_dirs:
            print("No chunk directories found")
            return None
            
        chunk_dir = chunk_dirs[0]  # Usually chunk-000
        
        # Get all camera dirs
        camera_dirs = [d for d in chunk_dir.iterdir() if d.is_dir()]
        if not camera_dirs:
            print(f"No camera directories found in {chunk_dir}")
            return None
            
        # Create episode directories and copy videos
        for episode_idx in range(recorded_episodes):
            episode_dir = sequence_dir_path / f"episode_{episode_idx}"
            os.makedirs(episode_dir, exist_ok=True)
            
            for camera_dir in camera_dirs:
                camera_name = camera_dir.name.split(".")[-1]  # Extract cam_high, cam_left_wrist, etc.
                
                # Find the episode video file
                episode_file = None
                for file in camera_dir.iterdir():
                    if file.is_file() and file.name.endswith(".mp4") and f"episode_{episode_idx:06d}" in file.name:
                        episode_file = file
                        break
                
                if episode_file:
                    # Copy to new structure with renamed file
                    dest_file = episode_dir / f"{camera_name}.mp4"
                    shutil.copy2(episode_file, dest_file)
                    print(f"Copied {camera_name} video for episode {episode_idx}")
                else:
                    print(f"Video file not found for episode {episode_idx} in {camera_dir}")
        
        print(f"Videos reorganized to: {sequence_dir_path}")
        return sequence_dir_path
            
    except Exception as e:
        print(f"Error reorganizing videos: {str(e)}")
        return None

########################################################################################
# Control modes
########################################################################################


@safe_disconnect
def calibrate(robot: Robot, cfg: CalibrateControlConfig):
    # TODO(aliberts): move this code in robots' classes
    if robot.robot_type.startswith("stretch"):
        if not robot.is_connected:
            robot.connect()
        if not robot.is_homed():
            robot.home()
        return

    arms = robot.available_arms if cfg.arms is None else cfg.arms
    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]
    available_arms_str = " ".join(robot.available_arms)
    unknown_arms_str = " ".join(unknown_arms)

    if arms is None or len(arms) == 0:
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    if len(unknown_arms) > 0:
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    for arm_id in arms:
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"
        if arm_calib_path.exists():
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            print(f"Calibration file not found '{arm_calib_path}'")

    if robot.is_connected:
        robot.disconnect()

    # Calling `connect` automatically runs calibration
    # when the calibration file is missing
    robot.connect()
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")


@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,
        display_cameras=cfg.display_cameras,
    )


@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    # TODO(rcadene): Add option to record logs
    
    # If root is provided, append a timestamp to ensure a unique folder is created each time
    if cfg.root and not cfg.resume:
        from datetime import datetime
        from pathlib import Path
        
        # Create a unique subfolder based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(cfg.root, str):
            cfg.root = str(Path(cfg.root).parent / f"{Path(cfg.root).name}_{timestamp}")
        else:  # Path object
            cfg.root = cfg.root.parent / f"{cfg.root.name}_{timestamp}"
        print(f"Creating new recording directory: {cfg.root}")
    
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
            local_files_only=cfg.local_files_only,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, cfg.device, ds_meta=dataset.meta)

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    log_say("Warmup record", cfg.play_sounds)
    warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_cameras, cfg.fps)

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    recorded_episodes = 0
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        record_episode(
            dataset=dataset,
            robot=robot,
            events=events,
            episode_time_s=cfg.episode_time_s,
            display_cameras=cfg.display_cameras,
            policy=policy,
            device=cfg.device,
            use_amp=cfg.use_amp,
            fps=cfg.fps,
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)
            reset_environment(robot, events, cfg.reset_time_s)

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode(cfg.single_task)
        recorded_episodes += 1

        if events["stop_recording"]:
            break

    log_say("Stop recording", cfg.play_sounds, blocking=True)
    stop_recording(robot, listener, cfg.display_cameras)

    if cfg.run_compute_stats:
        logging.info("Computing dataset statistics")

    dataset.consolidate(cfg.run_compute_stats)

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)
    
    # Reorganize videos into the requested format
    log_say("Reorganizing videos", cfg.play_sounds)
    sequence_dir = reorganize_videos(cfg.root, recorded_episodes)
    
    if sequence_dir:
        log_say(f"Videos reorganized to {sequence_dir}", cfg.play_sounds)

    log_say("Exiting", cfg.play_sounds)
    return dataset


@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    # TODO(rcadene, aliberts): refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO(rcadene): Add option to record logs

    dataset = LeRobotDataset(
        cfg.repo_id, root=cfg.root, episodes=[cfg.episode], local_files_only=cfg.local_files_only
    )
    actions = dataset.hf_dataset.select_columns("action")

    if not robot.is_connected:
        robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)


@safe_disconnect
def record_depth(robot: Robot, cfg: RecordDepthControlConfig):
    """
    Record depth and RGB images from RealSense cameras along with robot state.
    Saves:
    - Colorized depth maps
    - RGB images
    - Robot state data
    - Metadata in parquet format
    """
    if not robot.is_connected:
        robot.connect()

    # Create timestamp for this recording session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(cfg.root) / f"recordings_{timestamp}"
    
    # Create directories for each camera
    for camera_name in robot.cameras:
        camera_dir = session_dir / f"episode_000001" / "observation.images" / camera_name
        camera_dir.mkdir(parents=True, exist_ok=True)
        
        rgb_dir = camera_dir / "rgb"
        depth_dir = camera_dir / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRecording {cfg.num_frames} frames with {len(robot.cameras)} cameras...")
    print(f"Saving to: {session_dir}")

    metadata = []
    robot_states = []
    
    for frame_idx in range(cfg.num_frames):
        print(f"\nCapturing frame {frame_idx + 1}/{cfg.num_frames}")
        
        frame_data = {
            'frame_idx': frame_idx,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Get robot state
        try:
            robot_state = robot.get_state()
            robot_states.append({
                'frame_idx': frame_idx,
                'timestamp': frame_data['timestamp'],
                **robot_state
            })
        except Exception as e:
            print(f"Warning: Could not get robot state: {str(e)}")
        
        # Get frames from all cameras
        for camera_name, camera in robot.cameras.items():
            # Get camera directory paths
            camera_dir = session_dir / f"episode_000001" / "observation.images" / camera_name
            rgb_dir = camera_dir / "rgb"
            depth_dir = camera_dir / "depth"
            
            try:
                # Get both RGB and depth frames
                frames = camera.get_frames()
                if frames is None:
                    print(f"Warning: No frames received from {camera_name}")
                    continue
                
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    print(f"Warning: Invalid frames from {camera_name}")
                    continue
                
                # Convert frames to numpy arrays
                rgb_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Create colorized depth visualization
                colorizer = camera.colorizer
                colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                
                # Save RGB frame
                rgb_path = rgb_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                frame_data[f'{camera_name}_rgb_path'] = str(rgb_path)
                
                # Save raw depth data as numpy array
                depth_raw_path = depth_dir / f"frame_{frame_idx:06d}_depth.npy"
                np.save(str(depth_raw_path), depth_image)
                frame_data[f'{camera_name}_depth_raw_path'] = str(depth_raw_path)
                
                # Save colorized depth visualization
                depth_viz_path = depth_dir / f"frame_{frame_idx:06d}_depth_viz.png"
                cv2.imwrite(str(depth_viz_path), colorized_depth)
                frame_data[f'{camera_name}_depth_viz_path'] = str(depth_viz_path)
                
                print(f"Saved frames from {camera_name}")
                
            except Exception as e:
                print(f"Error capturing frames from {camera_name}: {str(e)}")
                continue
        
        metadata.append(frame_data)
        
        if frame_idx < cfg.num_frames - 1:
            print(f"Waiting {cfg.delay_between_frames} seconds before next capture...")
            time.sleep(cfg.delay_between_frames)

    # Save metadata as parquet
    metadata_df = pd.DataFrame(metadata)
    metadata_path = session_dir / f"episode_000001" / "metadata.parquet"
    metadata_df.to_parquet(metadata_path)
    print(f"\nSaved metadata to: {metadata_path}")

    # Save robot states as parquet
    if robot_states:
        robot_states_df = pd.DataFrame(robot_states)
        robot_states_path = session_dir / f"episode_000001" / "robot_states.parquet"
        robot_states_df.to_parquet(robot_states_path)
        print(f"Saved robot states to: {robot_states_path}")

    print("\nRecording complete!")
    return session_dir


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    if isinstance(cfg.control, CalibrateControlConfig):
        calibrate(robot, cfg.control)
    elif isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        record(robot, cfg.control)
    elif isinstance(cfg.control, ReplayControlConfig):
        replay(robot, cfg.control)
    elif isinstance(cfg.control, RecordDepthControlConfig):
        record_depth(robot, cfg.control)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_robot()
