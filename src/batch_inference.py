#!/usr/bin/env python3
"""
Batch inference script for SpaTrackerV2
Processes all videos in a directory using inference.py

Usage:
    python batch_inference.py <data_dir> [--fps FPS] [--data_type TYPE]

Example:
    python batch_inference.py /mnt/c/workspace/data/my_videos --fps 10
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}

def find_videos(data_dir):
    """Find all video files in the directory"""
    videos = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Directory '{data_dir}' does not exist")
        return videos

    for video_path in data_path.iterdir():
        if video_path.is_file() and video_path.suffix in VIDEO_EXTENSIONS:
            videos.append(video_path)

    return sorted(videos)

def run_inference(data_dir, video_name, fps=10, data_type="RGB"):
    """Run inference.py on a single video"""
    cmd = [
        "python3", "inference.py",
        f"--data_type={data_type}",
        f"--data_dir={data_dir}",
        f"--video_name={video_name}",
        f"--fps={fps}"
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Process returned non-zero exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for SpaTrackerV2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_inference.py /mnt/c/workspace/data/videos
  python batch_inference.py /mnt/c/workspace/data/videos --fps 15
  python batch_inference.py ./my_videos --fps 10 --data_type RGB
        """
    )

    parser.add_argument('data_dir', type=str,
                        help='Directory containing video files')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frame sampling rate (default: 10)')
    parser.add_argument('--data_type', type=str, default='RGB',
                        choices=['RGB', 'RGBD'],
                        help='Data type: RGB or RGBD (default: RGB)')

    args = parser.parse_args()

    # Find all videos
    videos = find_videos(args.data_dir)

    if not videos:
        print(f"\nNo video files found in '{args.data_dir}'")
        print(f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}")
        return 1

    # Print summary
    print("=" * 60)
    print("SpaTrackerV2 Batch Inference")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Data type:      {args.data_type}")
    print(f"FPS:            {args.fps}")
    print(f"Videos found:   {len(videos)}")
    print("=" * 60)
    print()

    # Process each video
    processed = 0
    failed = 0

    for i, video_path in enumerate(videos, 1):
        video_name = video_path.stem  # Filename without extension

        print("━" * 60)
        print(f"[{i}/{len(videos)}] Processing: {video_path.name}")
        print(f"Video name: {video_name}")
        print("━" * 60)

        success = run_inference(
            args.data_dir,
            video_name,
            fps=args.fps,
            data_type=args.data_type
        )

        if success:
            print(f"✓ Successfully processed: {video_path.name}")
            processed += 1
        else:
            print(f"✗ Failed to process: {video_path.name}")
            failed += 1

        print()

    # Print final summary
    print("=" * 60)
    print("Batch Inference Complete")
    print("=" * 60)
    print(f"Successfully processed: {processed} videos")
    print(f"Failed:                 {failed} videos")
    print("=" * 60)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
