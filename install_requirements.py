#!/usr/bin/env python3
"""
Script to install requirements in the correct order to avoid dependency conflicts.
This resolves the issues with daal4py, torchtext, and numba compatibility.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def main():
    """Install requirements in the correct order."""
    print("Installing requirements in the correct order to avoid dependency conflicts...")
    
    # Step 1: Install PyTorch first with specific CUDA version
    torch_command = "pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118"
    if not run_command(torch_command, "Installing PyTorch with CUDA 11.8 support"):
        print("Failed to install PyTorch. Please check your CUDA installation.")
        return False
    
    # Step 2: Install numpy with specific version
    numpy_command = "pip install numpy==1.24.4"
    if not run_command(numpy_command, "Installing numpy 1.24.4"):
        return False
    
    # Step 3: Install other core dependencies
    core_deps = [
        "scenedetect[opencv]>=0.6.2",
        "opencv-python>=4.8.0", 
        "ffmpeg-python>=0.2.0",
        "faiss-cpu>=1.7.4",
        "pandas>=1.5.0",
        "pillow>=9.0.0",
        "python-dotenv>=1.0.0",
        "openai>=1.0.0"
    ]
    
    for dep in core_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    # Step 4: Install CLIP
    clip_command = "pip install git+https://github.com/openai/CLIP.git"
    if not run_command(clip_command, "Installing CLIP"):
        return False
    
    print("\n✓ All requirements installed successfully!")
    print("\nNote: If you still encounter issues, try:")
    print("1. Create a fresh virtual environment")
    print("2. Upgrade pip: pip install --upgrade pip")
    print("3. Run this script again")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
