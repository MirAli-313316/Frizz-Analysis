#!/usr/bin/env python3
"""
Simple test script for BiRefNet integration.

Tests the basic functionality of the BiRefNet hair frizz analysis pipeline.
"""

import sys
import time
from pathlib import Path

try:
    import cv2
    import numpy as np

    # Import our modules (absolute imports)
    from src.analysis import analyze_image

except ImportError as e:
    print(f"ERROR: Missing required modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Main test function."""
    print("BIREFNET INTEGRATION TEST")
    print("=" * 60)
    print("Testing the BiRefNet hair frizz analysis pipeline")
    print("=" * 60)

    # Determine test image
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "test_images/IMG_8787.JPG"

    print(f"Using test image: {test_image}")
    print()

    # Create output directory
    Path("outputs/test_birefnet").mkdir(parents=True, exist_ok=True)

    # Test complete pipeline
    print("Running complete analysis pipeline...")
    try:
        start_time = time.time()

        # Run complete analysis
        result = analyze_image(
            test_image,
            visualize=True,
            output_dir="outputs/test_birefnet",
            num_expected_tresses=6
        )

        total_time = time.time() - start_time

        print("SUCCESS: Complete analysis completed!")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Total surface area: {result.get_total_area():.2f} cm²")
        print(f"   Number of tresses: {len(result.tresses)}")
        print(f"   Calibration factor: {result.calibration_factor:.6f} cm²/pixel")
        print(f"   Device used: {result.device_used}")
        print(f"   Processing time: {result.processing_time:.2f}s")

        # Show per-tress results
        print("   Per-tress surface areas:")
        for tress in result.tresses:
            print(f"     Tress {tress.tress_id}: {tress.area_cm2:.2f} cm²")

        # Check for output files
        output_dir = Path("outputs/test_birefnet")
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            print(f"   Generated {len(files)} output files:")
            for file in files:
                print(f"     - {file.name}")

        print()
        print("CONCLUSION: BiRefNet integration is working correctly!")
        print("   You can now run the full application with: python run_gui.py")
        return 0

    except Exception as e:
        print(f"FAILED: Pipeline test failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
