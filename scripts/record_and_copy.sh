#!/bin/bash

# Configuration
ROBOT_USER="johnny_one"
ROBOT_HOST="10.1.10.178"  # Update this with your robot PC's hostname or IP
ROBOT_RECORDINGS_PATH="/home/johnny_one/lerobot/recordings"
LOCAL_RECORDINGS_PATH="/Users/shreyas.chakravarthula/scale/lerobot/recordings"
ROBOT_LEROBOT_PATH="/home/johnny_one/lerobot"  # Updated to correct path
ROBOT_PASSWORD="scale"  # WARNING: Storing passwords in scripts is insecure!

# Default values
TASK_DESCRIPTION="Grasp a lego block and put it in the bin."
NUM_EPISODES=1
EPISODE_TIME=300
RESET_TIME=60
WARMUP_TIME=15
FPS=30

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --task=*)
      TASK_DESCRIPTION="${1#*=}"
      ;;
    --episodes=*)
      NUM_EPISODES="${1#*=}"
      ;;
    --episode-time=*)
      EPISODE_TIME="${1#*=}"
      ;;
    --reset-time=*)
      RESET_TIME="${1#*=}"
      ;;
    --warmup-time=*)
      WARMUP_TIME="${1#*=}"
      ;;
    --fps=*)
      FPS="${1#*=}"
      ;;
    --password=*)
      ROBOT_PASSWORD="${1#*=}"
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
  shift
done

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo "Error: sshpass is not installed. Please install it with: brew install sshpass"
    exit 1
fi

# Create timestamp for unique folder name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_FOLDER_NAME="recording_${TIMESTAMP}"
ROBOT_RECORDING_DIR="${ROBOT_RECORDINGS_PATH}/${BASE_FOLDER_NAME}"

echo "===== Starting recording session ====="
echo "Task: $TASK_DESCRIPTION"
echo "Episodes: $NUM_EPISODES"
echo "Episode Time: $EPISODE_TIME seconds"
echo "Reset Time: $RESET_TIME seconds"
echo "Warmup Time: $WARMUP_TIME seconds"
echo "FPS: $FPS"
echo "Recording directory will be: $ROBOT_RECORDING_DIR"

# Escape the task description for command line use
ESCAPED_TASK=$(echo "$TASK_DESCRIPTION" | sed 's/"/\\"/g')

# Create the record command
RECORD_CMD="cd $ROBOT_LEROBOT_PATH && \
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate lerobot && \
    python lerobot/scripts/control_robot.py \
    --robot.type=trossen_ai_bimanual \
    --robot.max_relative_target=null \
    --control.type=record \
    --control.fps=$FPS \
    --control.single_task=\"$ESCAPED_TASK\" \
    --control.repo_id=lerobot/trossen_ai_bimanual_test \
    --control.root=$ROBOT_RECORDING_DIR \
    --control.warmup_time_s=$WARMUP_TIME \
    --control.episode_time_s=$EPISODE_TIME \
    --control.reset_time_s=$RESET_TIME \
    --control.num_episodes=$NUM_EPISODES \
    --control.push_to_hub=false"

# Run the recording command on the robot PC using sshpass
echo "Running recording command on robot PC..."
sshpass -p "$ROBOT_PASSWORD" ssh "$ROBOT_USER@$ROBOT_HOST" "$RECORD_CMD"

if [ $? -ne 0 ]; then
    echo "Error: Recording command failed"
    exit 1
fi

echo "Recording completed."

# Find the actual recording directory (it has timestamp appended)
echo "Finding the exact recording directory..."
ACTUAL_RECORDING_DIR=$(sshpass -p "$ROBOT_PASSWORD" ssh "$ROBOT_USER@$ROBOT_HOST" "find $ROBOT_RECORDINGS_PATH -maxdepth 1 -type d -name \"${BASE_FOLDER_NAME}*\" | sort -r | head -n 1")

if [ -z "$ACTUAL_RECORDING_DIR" ]; then
    echo "Error: Could not find the recording directory"
    exit 1
fi

echo "Found recording directory: $ACTUAL_RECORDING_DIR"

# Create a unique sequence name with timestamp for robot PC
ROBOT_SEQUENCE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROBOT_SEQUENCE_DIR="${ROBOT_RECORDINGS_PATH}/sequence-${ROBOT_SEQUENCE_TIMESTAMP}"

# Create script to reorganize files on the robot PC
REORGANIZE_SCRIPT="
mkdir -p '$ROBOT_SEQUENCE_DIR'
echo 'Creating reorganized videos in $ROBOT_SEQUENCE_DIR'

# Find the first chunk directory (usually chunk-000)
CHUNK_DIR=\$(find '$ACTUAL_RECORDING_DIR/videos' -maxdepth 1 -type d -name 'chunk-*' | head -n 1)
if [ -z \"\$CHUNK_DIR\" ]; then
    echo 'Error: Could not find chunk directory'
    exit 1
fi

# Process each episode
for (( i=0; i<$NUM_EPISODES; i++ )); do
    PADDED_IDX=\$(printf '%06d' \$i)
    EPISODE_DIR='$ROBOT_SEQUENCE_DIR/episode_'\$i
    mkdir -p \"\$EPISODE_DIR\"
    echo \"Processing episode \$i in \$EPISODE_DIR\"
    
    # Process each camera
    for CAM_DIR in \"\$CHUNK_DIR\"/*; do
        if [ -d \"\$CAM_DIR\" ]; then
            # Extract camera name (e.g., cam_high, cam_left_wrist)
            CAM_NAME=\$(basename \"\$CAM_DIR\" | awk -F'.' '{print \$NF}')
            
            # Find episode file
            EPISODE_FILE=\$(find \"\$CAM_DIR\" -name \"episode_\${PADDED_IDX}.mp4\")
            
            if [ -n \"\$EPISODE_FILE\" ]; then
                # Copy to destination with simplified name
                cp \"\$EPISODE_FILE\" \"\${EPISODE_DIR}/\${CAM_NAME}.mp4\"
                echo \"Copied \${CAM_NAME} video for episode \$i\"
            else
                echo \"Warning: No video found for \${CAM_NAME} in episode \$i\"
            fi
        fi
    done
done

echo 'Reorganization complete on robot PC'
echo \"Videos saved in: $ROBOT_SEQUENCE_DIR\"
"

# Run the reorganization script on the robot PC
echo "Reorganizing videos on the robot PC..."
sshpass -p "$ROBOT_PASSWORD" ssh "$ROBOT_USER@$ROBOT_HOST" "$REORGANIZE_SCRIPT"

# Create a temporary directory for the raw files on local Mac
LOCAL_TEMP_DIR="${LOCAL_RECORDINGS_PATH}/temp_${TIMESTAMP}"
mkdir -p "$LOCAL_TEMP_DIR"

# Copy the data from robot to local Mac using sshpass
echo "Copying data from robot PC to local Mac..."
sshpass -p "$ROBOT_PASSWORD" scp -r "$ROBOT_USER@$ROBOT_HOST:$ACTUAL_RECORDING_DIR/videos" "$LOCAL_TEMP_DIR/"

if [ $? -ne 0 ]; then
    echo "Error: SCP command failed"
    exit 1
fi

echo "Data copied to temporary directory. Reorganizing on local Mac..."

# Create the final directory structure (sequence-timestamp) on local Mac
SEQUENCE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SEQUENCE_DIR="${LOCAL_RECORDINGS_PATH}/sequence-${SEQUENCE_TIMESTAMP}"
mkdir -p "$SEQUENCE_DIR"

# Reorganize videos into the requested structure on local Mac
for ((i=0; i<NUM_EPISODES; i++)); do
    PADDED_IDX=$(printf "%06d" $i)
    EPISODE_DIR="${SEQUENCE_DIR}/episode_${i}"
    mkdir -p "$EPISODE_DIR"
    
    # Find the chunk directory (usually chunk-000)
    CHUNK_DIR=$(find "$LOCAL_TEMP_DIR/videos" -maxdepth 1 -type d -name "chunk-*" | head -n 1)
    
    if [ -z "$CHUNK_DIR" ]; then
        echo "Error: Could not find chunk directory"
        exit 1
    fi
    
    # Process each camera
    for CAM_DIR in "$CHUNK_DIR"/*; do
        if [ -d "$CAM_DIR" ]; then
            # Extract camera name (e.g., cam_high, cam_left_wrist)
            CAM_NAME=$(basename "$CAM_DIR" | awk -F'.' '{print $NF}')
            
            # Find episode file
            EPISODE_FILE=$(find "$CAM_DIR" -name "episode_${PADDED_IDX}.mp4")
            
            if [ -n "$EPISODE_FILE" ]; then
                # Copy to destination with simplified name
                cp "$EPISODE_FILE" "${EPISODE_DIR}/${CAM_NAME}.mp4"
                echo "Copied ${CAM_NAME} video for episode ${i}"
            else
                echo "Warning: No video found for ${CAM_NAME} in episode ${i}"
            fi
        fi
    done
done

# Clean up temporary directory
rm -rf "$LOCAL_TEMP_DIR"

echo "===== Recording and copying completed successfully ====="
echo "Data is available in 3 places:"
echo "1. Original format on robot PC: $ACTUAL_RECORDING_DIR"
echo "2. Reorganized format on robot PC: $ROBOT_SEQUENCE_DIR"
echo "3. Reorganized format on local Mac: $SEQUENCE_DIR"
echo
echo "Local directory structure:"
echo "  $SEQUENCE_DIR/"
echo "    episode_0/"
echo "      cam_high.mp4"
echo "      cam_left_wrist.mp4"
echo "      cam_low.mp4"
echo "      cam_right_wrist.mp4"
echo "    episode_1/"
echo "      ..."
echo "    ..." 