"""
Helper to install pycolmap compatible with CUDA 12.8
"""

import subprocess
import sys

print("Installing pycolmap for CUDA 12.8...")
print()

# First, uninstall existing pycolmap
print("Step 1: Removing existing pycolmap...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pycolmap"])

print("\nStep 2: Installing pycolmap from source (will compile with your CUDA 12.8)...")
print("This may take several minutes...\n")

# Install from source - will use system CUDA
result = subprocess.run([
    sys.executable, "-m", "pip", "install",
    "pycolmap",
    "--no-binary", "pycolmap",
    "-v"
])

if result.returncode == 0:
    print("\n✓ Installation successful!")
    print("\nTesting import...")
    try:
        import pycolmap
        print(f"✓ pycolmap {pycolmap.__version__} imported successfully!")
    except Exception as e:
        print(f"✗ Import failed: {e}")
else:
    print("\n✗ Installation failed")
    print("\nAlternative: Use conda-forge")
    print("  conda install -c conda-forge pycolmap")
