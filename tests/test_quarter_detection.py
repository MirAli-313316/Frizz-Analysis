#!/usr/bin/env python3
"""
Comprehensive test script for hybrid BiRefNet quarter detection with visual outputs.

Tests quarter detection and generates visual verification images showing:
- Detected quarter with red circle overlay
- Zoomed-in view of quarter region for detailed inspection
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.analysis import analyze_image
    from src.calibration import detect_quarter, visualize_calibration

except ImportError as e:
    print(f"ERROR: Missing required modules: {e}")
    sys.exit(1)


def test_quarter_detection_with_visuals(image_path: str = None, output_dir: str = "outputs/quarter_test"):
    """Test quarter detection and generate visual verification."""

    if image_path is None:
        image_path = 'test_images/IMG_8787.JPG'

    print(f"Testing quarter detection for: {image_path}")
    print("=" * 60)

    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    try:
        # Test quarter detection directly
        quarter_result = detect_quarter(image_path)

        print("SUCCESS: Quarter detection successful!")
        print(f"  Center: ({quarter_result.quarter_center[0]}, {quarter_result.quarter_center[1]})")
        print(f"  Radius: {quarter_result.quarter_radius}px")
        print(f"  Area: {quarter_result.quarter_area_pixels:.0f}px")
        print(f"  Calibration factor: {quarter_result.calibration_factor:.6f} cm²/pixel")
        print(f"  Confidence: {quarter_result.confidence:.3f}")

        # Generate calibration visualization
        viz_image = visualize_calibration(image_rgb, quarter_result)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main visualization with quarter detection
        main_viz_path = output_path / f"quarter_detection_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(main_viz_path), cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
        print(f"SUCCESS: Saved quarter visualization: {main_viz_path}")

        # Create zoomed-in view of quarter region for detailed inspection
        center_x, center_y = quarter_result.quarter_center
        radius = quarter_result.quarter_radius

        # Add padding for zoom view
        zoom_size = int(radius * 3)
        zoom_x1 = max(0, center_x - zoom_size)
        zoom_y1 = max(0, center_y - zoom_size)
        zoom_x2 = min(image.shape[1], center_x + zoom_size)
        zoom_y2 = min(image.shape[0], center_y + zoom_size)

        zoomed_region = image_rgb[zoom_y1:zoom_y2, zoom_x1:zoom_x2]

        # Draw quarter circle on zoomed region
        local_center_x = center_x - zoom_x1
        local_center_y = center_y - zoom_y1

        zoomed_with_circle = zoomed_region.copy()
        cv2.circle(zoomed_with_circle, (local_center_x, local_center_y), radius, (255, 0, 0), 3)
        cv2.circle(zoomed_with_circle, (local_center_x, local_center_y), 2, (255, 0, 0), -1)

        # Save zoomed visualization
        zoom_path = output_path / f"quarter_zoom_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(zoom_path), cv2.cvtColor(zoomed_with_circle, cv2.COLOR_RGB2BGR))
        print(f"SUCCESS: Saved zoomed quarter view: {zoom_path}")

        print(f"\nVISUAL OUTPUTS GENERATED:")
        print(f"   - {main_viz_path}")
        print(f"   - {zoom_path}")
        print(f"\nOpen these images to visually verify:")
        print(f"     - Red circle shows detected quarter location")
        print(f"     - Zoomed view helps inspect quarter details")
        print(f"     - Compare with your original image")

        return True

    except Exception as e:
        print(f"FAILED: Quarter detection failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")

        # Save error visualization showing search regions that were tested
        error_viz = image.copy()

        # Draw search regions that were attempted
        search_regions = [
            (0, 0, min(700, image.shape[1]), min(700, image.shape[0])),
            (0, 0, min(1000, image.shape[1]), min(1000, image.shape[0])),
            (0, 0, image.shape[1], min(700, image.shape[0])),
            (0, 0, min(800, image.shape[1]), min(800, image.shape[0])),
            (50, 50, min(600, image.shape[1]-100), min(600, image.shape[0]-100)),
            (0, 0, min(1200, image.shape[1]), min(800, image.shape[0])),
        ]

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, (x, y, w, h) in enumerate(search_regions):
            cv2.rectangle(error_viz, (x, y), (x + w, y + h), colors[i], 3)
            cv2.putText(error_viz, f"Region {i+1}", (x + 10, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

        error_path = Path(output_dir) / f"quarter_error_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(error_path), error_viz)
        print(f"SUCCESS: Saved error visualization: {error_path}")
        print(f"This shows all search regions that were tested")

        return False


def main():
    """Main test function with visual outputs."""
    print("HYBRID BIREFNET QUARTER DETECTION TEST WITH VISUAL OUTPUTS")
    print("=" * 70)
    print("Testing quarter detection and generating visual verification images")
    print("=" * 70)

    # Determine test image
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "test_images/IMG_8787.JPG"

    print(f"Using test image: {test_image}")
    print()

    success = test_quarter_detection_with_visuals(test_image)

    print("\n" + "=" * 70)
    if success:
        print("QUARTER DETECTION SUCCESSFUL!")
        print("Visual outputs have been generated")
        print("Check the 'outputs/quarter_test/' folder for:")
        print("   - quarter_detection_*.jpg (full image with red circle)")
        print("   - quarter_zoom_*.jpg (zoomed quarter view)")
        print("\nOpen these images to visually verify the quarter detection!")
    else:
        print("QUARTER DETECTION FAILED")
        print("Check the 'outputs/quarter_test/' folder for error visualization")
        print("This shows which regions were searched for the quarter.")

    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
