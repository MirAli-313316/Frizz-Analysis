"""
Launcher script for the Hair Frizz Analysis GUI.

Run this script to start the graphical interface:
    python run_gui.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run
from src.gui import main

if __name__ == "__main__":
    main()


