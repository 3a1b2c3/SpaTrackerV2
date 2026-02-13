"""
Check CUDA compatibility and GPU compute capability
"""

import torch

print("="*60)
print("CUDA & GPU Information")
print("="*60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"\nNumber of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")

    # Check what architectures PyTorch was compiled for
    print("\n" + "="*60)
    print("PyTorch CUDA architectures:")
    print("="*60)
    if hasattr(torch.cuda, 'get_arch_list'):
        archs = torch.cuda.get_arch_list()
        print(f"Compiled for: {archs}")

    # Try to identify the issue
    print("\n" + "="*60)
    print("Diagnosis:")
    print("="*60)

    gpu_arch = f"{props.major}.{props.minor}"
    print(f"\nYour GPU compute capability: {gpu_arch}")

    if hasattr(torch.cuda, 'get_arch_list'):
        archs = torch.cuda.get_arch_list()
        # Convert to comparable format
        arch_versions = [a.replace('sm_', '').replace('compute_', '') for a in archs]
        gpu_arch_num = int(gpu_arch.replace('.', ''))

        print(f"PyTorch compiled for: {', '.join(archs)}")

        # Check if compatible
        compatible = any(int(a) <= gpu_arch_num for a in arch_versions if a.isdigit())

        if not compatible:
            print("\n⚠ MISMATCH DETECTED!")
            print(f"Your GPU (sm_{gpu_arch.replace('.', '')}) is not in the compiled architectures.")
            print("\nSOLUTION:")
            print("Reinstall PyTorch with correct CUDA version:")
            print(f"  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu{torch.version.cuda.replace('.', '')}")
else:
    print("\n✗ CUDA not available!")
    print("Check:")
    print("1. NVIDIA drivers installed")
    print("2. CUDA toolkit installed")
    print("3. WSL2 CUDA support enabled")

print("\n" + "="*60)
print("Quick Fix:")
print("="*60)
print("\nReinstall PyTorch matching your system:")
print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
print("\nOr for CUDA 12.8:")
print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
