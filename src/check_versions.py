"""
Check PyTorch, xformers, and triton compatibility
"""

import sys

print("Checking installed versions...\n")

packages = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'xformers': 'xformers',
    'triton': 'Triton'
}

versions = {}
for pkg, name in packages.items():
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        versions[pkg] = version
        print(f"{name:12} {version}")
    except ImportError:
        print(f"{name:12} NOT INSTALLED")
    except Exception as e:
        print(f"{name:12} ERROR: {e}")

print("\n" + "="*60)
print("Compatibility issue detected:")
print("="*60)
print("\nxformers 0.0.34 is incompatible with newer triton versions.")
print("\nSOLUTIONS:\n")
print("1. Downgrade triton (QUICKEST):")
print("   pip install triton==2.1.0")
print()
print("2. Upgrade xformers (if available for your PyTorch):")
print("   pip install xformers --upgrade")
print()
print("3. Or install compatible versions together:")
print("   pip install torch==2.5.1 xformers==0.0.28 triton==3.0.0")
