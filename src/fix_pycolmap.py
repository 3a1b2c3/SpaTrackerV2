"""
Fix corrupted pycolmap installation and reinstall clean
"""

import os
import shutil
import sys

site_packages = r"C:\Users\kschmid\AppData\Local\Programs\Python\Python312\Lib\site-packages"

print("Cleaning up corrupted pycolmap installations...")
print(f"Site-packages: {site_packages}\n")

# List of corrupted directories to remove
corrupted = [
    "~ycolmap",
    "~~colmap",
]

removed = []
for item in corrupted:
    full_path = os.path.join(site_packages, item)
    if os.path.exists(full_path):
        print(f"Removing: {full_path}")
        try:
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
            removed.append(item)
            print(f"  ✓ Removed")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print(f"Not found: {full_path}")

# Also look for any pycolmap-related dist-info directories
print("\nLooking for pycolmap dist-info directories...")
for item in os.listdir(site_packages):
    if 'pycolmap' in item.lower() or 'colmap' in item.lower():
        full_path = os.path.join(site_packages, item)
        print(f"Found: {item}")
        if item.startswith('~'):
            print(f"  Removing corrupted: {item}")
            try:
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                else:
                    os.remove(full_path)
                print(f"  ✓ Removed")
            except Exception as e:
                print(f"  ✗ Error: {e}")

print("\n" + "="*60)
print("Now installing pycolmap fresh...")
print("="*60 + "\n")

import subprocess

# Install latest version
result = subprocess.run([
    sys.executable, "-m", "pip", "install",
    "pycolmap==3.13.0",
    "--force-reinstall"
], capture_output=False)

if result.returncode == 0:
    print("\n✓ Installation successful!")
    print("\nTesting import...")
    try:
        import pycolmap
        print(f"✓ pycolmap {pycolmap.__version__} works!")
    except Exception as e:
        print(f"✗ Import failed: {e}")
else:
    print("\n✗ Installation failed")
