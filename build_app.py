#!/usr/bin/env python3
"""
Build script for creating desktop application with PyInstaller.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, shell=False):
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"OK {command if isinstance(command, str) else ' '.join(command)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED Command failed: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Build the desktop application."""
    print("Building Frizz Analysis Desktop Application")
    print("=" * 50)

    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print("OK PyInstaller is available")
    except ImportError:
        print("Installing PyInstaller...")
        if not run_command([sys.executable, "-m", "pip", "install", "pyinstaller"]):
            print("FAILED Failed to install PyInstaller")
            return 1

    # Install the package in development mode first
    print("\nInstalling package in development mode...")
    if not run_command([sys.executable, "-m", "pip", "install", "-e", "."]):
        print("FAILED Failed to install package")
        return 1

    # Build with PyInstaller
    print("\nBuilding executable with PyInstaller...")
    if not run_command([sys.executable, "-m", "PyInstaller", "app.spec", "--noconfirm"]):
        print("FAILED Failed to build application")
        return 1

    # Check what was created
    dist_dir = Path("dist")
    if dist_dir.exists():
        print("\nBuild outputs:")
        for item in dist_dir.iterdir():
            if item.is_dir():
                print(f"  {item.name}/")
                # List contents of first level directories
                for subitem in item.iterdir():
                    if subitem.is_file() and not subitem.name.startswith('.'):
                        size_mb = subitem.stat().st_size / (1024 * 1024)
                        print(f"    {subitem.name} ({size_mb:.1f} MB)")

    print("\nBuild complete!")
    print("\nTo run the application:")
    if os.name == 'nt':  # Windows
        print("  Double-click: dist/FrizzAnalysis.exe")
    elif sys.platform == 'darwin':  # macOS
        print("  Open: dist/Frizz Analysis.app")
    else:  # Linux
        print("  Run: dist/FrizzAnalysis")

    return 0

if __name__ == "__main__":
    sys.exit(main())
