"""
Minimal test for pycolmap installation and basic functionality
"""

try:
    import pycolmap
    print(f"✓ pycolmap imported successfully")
    print(f"  Version: {pycolmap.__version__}")

    # Test basic camera creation
    camera = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=640,
        height=480,
        params=[500, 320, 240]  # focal_length, cx, cy
    )
    print(f"✓ Camera created: {camera.model_name}, {camera.width}x{camera.height}")

    # Test creating a reconstruction object
    reconstruction = pycolmap.Reconstruction()
    print(f"✓ Reconstruction object created")
    print(f"  Cameras: {reconstruction.num_cameras()}")
    print(f"  Images: {reconstruction.num_images()}")
    print(f"  Points: {reconstruction.num_points3D()}")

    print("\n✓ All pycolmap tests passed!")

except ImportError as e:
    print(f"✗ Failed to import pycolmap: {e}")
    print("  Install with: pip install pycolmap")

except Exception as e:
    print(f"✗ Error during pycolmap test: {e}")
