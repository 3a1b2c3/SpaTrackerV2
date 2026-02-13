"""
Download and install pycolmap wheel directly
"""

import subprocess
import sys
import urllib.request
import json
import platform

print("Finding pycolmap wheel from PyPI...")

try:
    # Get PyPI package info
    pypi_url = "https://pypi.org/pypi/pycolmap/json"
    with urllib.request.urlopen(pypi_url) as response:
        data = json.loads(response.read())

    # Get Python version info
    py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Map platform to wheel platform tag
    if system == "windows":
        if machine == "amd64" or machine == "x86_64":
            platform_tag = "win_amd64"
        else:
            platform_tag = "win32"
    elif system == "linux":
        platform_tag = "manylinux"  # Will match manylinux wheels
    elif system == "darwin":
        platform_tag = "macosx"
    else:
        platform_tag = system

    print(f"Looking for: {py_version}, {platform_tag}\n")

    # Find matching wheel
    wheel_url = None
    wheel_filename = None

    for url, file_info in data["urls"].items():
        filename = file_info.get("filename", "")
        if filename.endswith(".whl") and py_version in filename and platform_tag in filename:
            wheel_url = file_info["url"]
            wheel_filename = filename
            break

    if not wheel_url:
        raise Exception(f"No matching wheel found for {py_version} on {platform_tag}")

    print(f"Found: {wheel_filename}")
    print(f"URL: {wheel_url}\n")

    print("Downloading...")
    urllib.request.urlretrieve(wheel_url, wheel_filename)
    print("✓ Downloaded\n")

    print("Installing...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        wheel_filename,
        "--force-reinstall"
    ])

    if result.returncode == 0:
        print("\n✓ Installed!")
        print("\nTesting...")
        import pycolmap
        print(f"✓ SUCCESS: pycolmap {pycolmap.__version__}")
    else:
        print("\n✗ Installation failed")

except Exception as e:
    print(f"✗ Error: {e}")
    print("\nTrying pip install method...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "pycolmap",
            "--force-reinstall"
        ])
        if result.returncode == 0:
            print("\n✓ Installed via pip!")
        else:
            print("\n✗ Pip install failed")
            print("\nFalling back to conda method:")
            print("  conda install -c conda-forge pycolmap")
    except Exception as e2:
        print(f"✗ Pip error: {e2}")
        print("\nFalling back to conda method:")
        print("  conda install -c conda-forge pycolmap")
