"""
Low-level DLL dependency checker for pycolmap
"""

import sys
import os
import torch

print("Environment Information:")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
print()

# Try to get more specific error info
import ctypes
import traceback

print("="*60)
print("Attempting detailed import...")
print("="*60)

try:
    # Try importing with traceback
    import pycolmap
    print("✓ SUCCESS! pycolmap imported")
    print(f"Version: {pycolmap.__version__}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}")
    print()
    traceback.print_exc()
    print()

    # Try to find pycolmap location
    try:
        import importlib.util
        spec = importlib.util.find_spec("pycolmap")
        if spec and spec.origin:
            print(f"pycolmap location: {spec.origin}")
            pyd_dir = os.path.dirname(spec.origin)
            print(f"Directory: {pyd_dir}")

            # List .pyd and .dll files
            if os.path.exists(pyd_dir):
                print("\nFiles in pycolmap directory:")
                for f in os.listdir(pyd_dir):
                    if f.endswith(('.pyd', '.dll')):
                        print(f"  - {f}")
    except:
        pass

    print("\n" + "="*60)
    print("SOLUTIONS:")
    print("="*60)
    print()
    print("Since you have PyTorch 2.10.0 installed, try:")
    print()
    print("1. Use conda-forge (RECOMMENDED for CUDA compatibility):")
    print("   conda install -c conda-forge pycolmap")
    print()
    print("2. Or install Visual C++ Redistributables:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print()
    print("3. Or try an older pycolmap version:")
    print("   pip uninstall pycolmap")
    print("   pip install pycolmap==0.6.1")
    print()
    print("4. Or build from source with your exact CUDA version:")
    print("   pip uninstall pycolmap")
    print("   pip install pycolmap --no-binary pycolmap")
