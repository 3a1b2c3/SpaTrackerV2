#!/bin/bash

# Batch inference script for SpaTrackerV2
# Usage: ./batch_inference.sh <data_dir> [fps]
#
# Example: ./batch_inference.sh /mnt/c/workspace/data/my_videos 10

if [ $# -lt 1 ]; then
    echo "Usage: $0 <data_dir> [fps]"
    echo "  data_dir: Directory containing video files"
    echo "  fps:      Frame sampling rate (default: 10)"
    exit 1
fi

DATA_DIR="$1"
FPS="${2:-10}"  # Default FPS is 10 if not provided

# Check if directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory '$DATA_DIR' does not exist"
    exit 1
fi

# Video file extensions to process
VIDEO_EXTENSIONS=("mp4" "avi" "mov" "mkv" "MP4" "AVI" "MOV" "MKV")

echo "========================================="
echo "SpaTrackerV2 Batch Inference"
echo "========================================="
echo "Data directory: $DATA_DIR"
echo "FPS: $FPS"
echo "========================================="
echo ""

# Counter for processed videos
PROCESSED=0
FAILED=0

# Find and process all video files
for ext in "${VIDEO_EXTENSIONS[@]}"; do
    for video_path in "$DATA_DIR"/*.$ext; do
        # Skip if no files match (glob didn't expand)
        [ -e "$video_path" ] || continue

        # Extract video filename without extension
        video_file=$(basename "$video_path")
        video_name="${video_file%.*}"

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Processing: $video_file"
        echo "Video name: $video_name"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Run inference
        python3 inference.py \
            --data_type="RGB" \
            --data_dir="$DATA_DIR" \
            --video_name="$video_name" \
            --fps=$FPS

        # Check if inference succeeded
        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed: $video_file"
            ((PROCESSED++))
        else
            echo "✗ Failed to process: $video_file"
            ((FAILED++))
        fi

        echo ""
    done
done

echo "========================================="
echo "Batch Inference Complete"
echo "========================================="
echo "Successfully processed: $PROCESSED videos"
echo "Failed: $FAILED videos"
echo "========================================="

if [ $PROCESSED -eq 0 ]; then
    echo "Warning: No video files found in $DATA_DIR"
    echo "Supported formats: ${VIDEO_EXTENSIONS[*]}"
fi
