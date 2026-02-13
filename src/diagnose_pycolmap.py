"""
Diagnose pycolmap DLL issues on Windows
"""

import sys
import os

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Check if pycolmap package exists
try:
    import pip
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "show", "pycolmap"],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("pycolmap package info:")
        print(result.stdout)
    else:
        print("✗ pycolmap not installed")
        print("  Install with: pip install pycolmap")
except Exception as e:
    print(f"Error checking pip: {e}")

print("\n" + "="*60)
print("Attempting to import pycolmap with detailed error...")
print("="*60)

try:
    import pycolmap
    print("✓ pycolmap imported successfully!")
    print(f"  Version: {pycolmap.__version__}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print()
    print("Common fixes for Windows DLL errors:")
    print("1. Install Visual C++ Redistributables:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print()
    print("2. Try installing from conda-forge (if using conda):")
    print("   conda install -c conda-forge pycolmap")
    print()
    print("3. Check CUDA compatibility (if using GPU):")
    print("   - pycolmap may require specific CUDA version")
    print("   - Ensure CUDA toolkit is installed")
    print()
    print("4. Try building from source or use a compatible wheel:")
    print("   pip uninstall pycolmap")
    print("   pip install pycolmap --no-cache-dir")

# Check for common DLL dependencies
print("\n" + "="*60)
print("Checking system paths and DLLs...")
print("="*60)

# Check if running in conda/virtual env
if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
    print(f"✓ Virtual environment detected: {sys.prefix}")
else:
    print("  Running in system Python")

# Check PATH for common locations
path_dirs = os.environ.get('PATH', '').split(os.pathsep)
print(f"\nPATH contains {len(path_dirs)} directories")

cuda_paths = [p for p in path_dirs if 'cuda' in p.lower()]
if cuda_paths:
    print(f"  CUDA paths found: {len(cuda_paths)}")
    for p in cuda_paths[:3]:
        print(f"    - {p}")
