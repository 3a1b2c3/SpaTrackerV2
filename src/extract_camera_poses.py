"""
Extract and visualize camera positions and extrinsics from SpatialTracker results
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os


def load_camera_data(npz_path):
    """Load camera data from NPZ file"""
    data = np.load(npz_path, allow_pickle=True)

    # Get extrinsics (world-to-camera)
    extrinsics = data["extrinsics"]  # [num_frames, 4, 4]
    intrinsics = data["intrinsics"]  # [num_frames, 3, 3]

    # Convert to camera-to-world
    c2w = np.linalg.inv(extrinsics)

    # Extract camera positions (translation)
    camera_positions = c2w[:, :3, 3]  # [num_frames, 3] - XYZ positions

    # Extract camera rotations
    camera_rotations = c2w[:, :3, :3]  # [num_frames, 3, 3]

    return {
        'extrinsics': extrinsics,
        'intrinsics': intrinsics,
        'c2w': c2w,
        'positions': camera_positions,
        'rotations': camera_rotations
    }


def print_camera_info(camera_data):
    """Print camera information"""
    positions = camera_data['positions']
    intrinsics = camera_data['intrinsics']

    print("=" * 60)
    print("CAMERA TRAJECTORY INFORMATION")
    print("=" * 60)
    print(f"\nNumber of frames: {len(positions)}")
    print(f"\nCamera positions (XYZ):")
    print(f"  Start position: {positions[0]}")
    print(f"  End position:   {positions[-1]}")
    print(f"  Min XYZ:        {positions.min(axis=0)}")
    print(f"  Max XYZ:        {positions.max(axis=0)}")
    print(f"  Total distance: {np.linalg.norm(positions[-1] - positions[0]):.3f}")

    print(f"\nIntrinsics (first frame):")
    print(f"  Focal length (fx, fy): ({intrinsics[0, 0, 0]:.2f}, {intrinsics[0, 1, 1]:.2f})")
    print(f"  Principal point (cx, cy): ({intrinsics[0, 0, 2]:.2f}, {intrinsics[0, 1, 2]:.2f})")
    print("=" * 60)


def visualize_camera_trajectory(camera_data, save_path=None):
    """Visualize camera trajectory in 3D"""
    positions = camera_data['positions']
    rotations = camera_data['rotations']

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2, label='Camera trajectory')

    # Plot camera positions as points
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=range(len(positions)), cmap='viridis', s=50, alpha=0.6)

    # Mark start and end
    ax.scatter(*positions[0], color='green', s=200, marker='o',
               label='Start', edgecolors='black', linewidths=2)
    ax.scatter(*positions[-1], color='red', s=200, marker='s',
               label='End', edgecolors='black', linewidths=2)

    # Draw camera coordinate frames (every N frames)
    step = max(1, len(positions) // 10)  # Show ~10 camera frames
    for i in range(0, len(positions), step):
        pos = positions[i]
        rot = rotations[i]

        # Camera axes (scaled for visibility)
        scale = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)) * 0.05

        # X-axis (right) - red
        ax.quiver(*pos, *rot[:, 0], color='red', length=scale,
                  arrow_length_ratio=0.3, alpha=0.6, linewidth=1.5)
        # Y-axis (down) - green
        ax.quiver(*pos, *rot[:, 1], color='green', length=scale,
                  arrow_length_ratio=0.3, alpha=0.6, linewidth=1.5)
        # Z-axis (forward) - blue
        ax.quiver(*pos, *rot[:, 2], color='blue', length=scale,
                  arrow_length_ratio=0.3, alpha=0.6, linewidth=1.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory and Poses', fontsize=14, fontweight='bold')
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.show()


def save_camera_poses_txt(camera_data, output_path):
    """Save camera poses to text file in standard format"""
    c2w = camera_data['c2w']

    with open(output_path, 'w') as f:
        f.write("# Camera-to-World Poses\n")
        f.write("# Format: frame_id, 4x4 transformation matrix (row-major)\n")
        f.write(f"# Total frames: {len(c2w)}\n\n")

        for i, pose in enumerate(c2w):
            f.write(f"Frame {i}:\n")
            for row in pose:
                f.write("  " + " ".join(f"{val:12.6f}" for val in row) + "\n")
            f.write("\n")

    print(f"Camera poses saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract camera positions and extrinsics from SpatialTracker results')
    parser.add_argument('npz_file', type=str, help='Path to the result NPZ file')
    parser.add_argument('--save_txt', type=str, default=None, help='Save poses to text file')
    parser.add_argument('--save_plot', type=str, default=None, help='Save visualization plot')
    parser.add_argument('--no_viz', action='store_true', help='Skip visualization')

    args = parser.parse_args()

    if not os.path.exists(args.npz_file):
        print(f"Error: File not found: {args.npz_file}")
        return

    # Load camera data
    print(f"Loading data from: {args.npz_file}")
    camera_data = load_camera_data(args.npz_file)

    # Print information
    print_camera_info(camera_data)

    # Save to text file if requested
    if args.save_txt:
        save_camera_poses_txt(camera_data, args.save_txt)

    # Visualize
    if not args.no_viz:
        visualize_camera_trajectory(camera_data, save_path=args.save_plot)


if __name__ == "__main__":
    main()
