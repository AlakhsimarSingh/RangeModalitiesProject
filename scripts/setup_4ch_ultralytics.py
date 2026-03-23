#!/usr/bin/env python3
"""
Setup script for 4-channel ultralytics modifications.

This script applies the 4-channel patches to a clean ultralytics installation,
enabling RGBD input support for YOLO training.

Usage:
    python scripts/setup_4ch_ultralytics.py

Requirements:
    - ultralytics==8.4.6 installed in current environment
    - git (for applying patches)
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command: {e}")
        return False


def find_ultralytics_path():
    """Find ultralytics installation path."""
    try:
        import ultralytics
        return Path(ultralytics.__file__).parent
    except ImportError:
        print("Error: ultralytics not found. Please install ultralytics==8.4.6 first.")
        return None


def apply_patch(ultralytics_path, patch_file):
    """Apply patch to ultralytics installation."""
    if not patch_file.exists():
        print(f"Error: Patch file not found: {patch_file}")
        return False

    print(f"Applying patch: {patch_file}")
    cmd = f'git apply "{patch_file}"'
    if not run_command(cmd, cwd=ultralytics_path):
        print("Failed to apply patch. Trying with --whitespace=fix...")
        cmd = f'git apply --whitespace=fix "{patch_file}"'
        if not run_command(cmd, cwd=ultralytics_path):
            print("Failed to apply patch with whitespace fix.")
            return False

    print("Patch applied successfully!")
    return True


def verify_modifications(ultralytics_path):
    """Verify that modifications were applied correctly."""
    print("Verifying modifications...")

    # Check base.py
    base_py = ultralytics_path / "data" / "base.py"
    if base_py.exists():
        with open(base_py, 'r') as f:
            content = f.read()
            if "cv2.IMREAD_UNCHANGED" in content and "channels == 4" in content:
                print("✓ base.py modifications verified")
            else:
                print("✗ base.py modifications not found")
                return False

    # Check loaders.py
    loaders_py = ultralytics_path / "data" / "loaders.py"
    if loaders_py.exists():
        with open(loaders_py, 'r') as f:
            content = f.read()
            count = content.count("cv2.IMREAD_UNCHANGED")
            if count >= 3:  # Should have 3 instances
                print(f"✓ loaders.py modifications verified ({count} instances)")
            else:
                print(f"✗ loaders.py modifications incomplete ({count} instances found)")
                return False

    # Check block.py
    block_py = ultralytics_path / "nn" / "modules" / "block.py"
    if block_py.exists():
        with open(block_py, 'r') as f:
            content = f.read()
            if "class Silence" in content and "Silence" in content:
                print("✓ block.py modifications verified")
            else:
                print("✗ block.py modifications not found")
                return False

    return True


def main():
    print("4-Channel Ultralytics Setup Script")
    print("=" * 40)

    # Find ultralytics path
    ultralytics_path = find_ultralytics_path()
    if not ultralytics_path:
        return 1

    print(f"Found ultralytics at: {ultralytics_path}")

    # Check if already modified
    if verify_modifications(ultralytics_path):
        print("4-channel modifications already applied!")
        return 0

    # Find patch file
    script_dir = Path(__file__).parent
    patch_file = script_dir.parent / "patches" / "ultralytics_4ch.patch"

    if not patch_file.exists():
        print(f"Error: Patch file not found at {patch_file}")
        print("Please ensure patches/ultralytics_4ch.patch exists.")
        return 1

    # Initialize git repo if needed
    git_dir = ultralytics_path / ".git"
    if not git_dir.exists():
        print("Initializing git repository in ultralytics...")
        if not run_command("git init", cwd=ultralytics_path):
            print("Failed to initialize git repository.")
            return 1
        if not run_command("git add .", cwd=ultralytics_path):
            print("Failed to add files to git.")
            return 1
        if not run_command('git commit -m "Clean ultralytics 8.4.6"', cwd=ultralytics_path):
            print("Failed to commit clean state.")
            return 1

    # Apply patch
    if not apply_patch(ultralytics_path, patch_file):
        return 1

    # Verify
    if not verify_modifications(ultralytics_path):
        print("Error: Modifications not applied correctly.")
        return 1

    print("\n✅ 4-channel ultralytics setup complete!")
    print("\nYou can now use 4-channel RGBD input with YOLO models.")
    print("Example data config:")
    print("  channels: 4")
    print("  nc: 1")
    print("  names: ['snowpole']")

    return 0


if __name__ == "__main__":
    sys.exit(main())