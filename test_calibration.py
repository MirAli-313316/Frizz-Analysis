"""
Test script for calibration module.

This script tests quarter detection on images in test_images/ directory
and displays visualizations to verify accuracy.
"""

import cv2
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from calibration import detect_quarter, create_visualization

def test_calibration():
    """Test calibration on all images in test_images directory."""
    
    test_dir = Path("test_images")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    if not test_dir.exists():
        print(f"Error: {test_dir} directory not found")
        return
    
    # Get all image files
    image_files = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return
    
    print(f"Found {len(image_files)} image(s) to test\n")
    print("=" * 80)
    
    results = []
    
    for image_path in sorted(image_files):
        print(f"\nTesting: {image_path.name}")
        print("-" * 80)
        
        try:
            # Detect quarter
            calibration = detect_quarter(str(image_path))
            
            print(f"[OK] Detection successful!")
            print(f"  Center: {calibration.quarter_center}")
            print(f"  Radius: {calibration.quarter_radius} pixels")
            print(f"  Quarter area: {calibration.quarter_area_pixels:.2f} pixels")
            print(f"  Calibration factor: {calibration.calibration_factor:.6f} cm²/pixel")
            print(f"  Confidence: {calibration.confidence:.1%}")
            
            # Create visualization
            vis_image = create_visualization(str(image_path), calibration)
            
            # Save visualization
            output_path = output_dir / f"calibration_{image_path.stem}.jpg"
            cv2.imwrite(str(output_path), vis_image)
            print(f"  Saved visualization: {output_path}")
            
            # Display (optional - comment out if running headless)
            # Resize for display if image is too large
            display_image = vis_image.copy()
            max_display_width = 1920
            height, width = display_image.shape[:2]
            if width > max_display_width:
                scale = max_display_width / width
                new_width = max_display_width
                new_height = int(height * scale)
                display_image = cv2.resize(display_image, (new_width, new_height))
            
            cv2.imshow(f"Calibration: {image_path.name}", display_image)
            print(f"\nVisualization displayed. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            results.append((image_path.name, calibration, True))
            
        except ValueError as e:
            print(f"[FAIL] Detection failed: {e}")
            results.append((image_path.name, None, False))
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            results.append((image_path.name, None, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for _, _, success in results if success)
    total = len(results)
    
    print(f"\nTotal images: {total}")
    print(f"Successful detections: {successful}")
    print(f"Failed detections: {total - successful}")
    
    if successful > 0:
        print("\nCalibration factors:")
        for name, calibration, success in results:
            if success and calibration:
                print(f"  {name}: {calibration.calibration_factor:.6f} cm²/pixel")
        
        # Check consistency
        factors = [cal.calibration_factor for _, cal, success in results if success and cal]
        if len(factors) > 1:
            import numpy as np
            mean_factor = np.mean(factors)
            std_factor = np.std(factors)
            cv_percent = (std_factor / mean_factor) * 100
            print(f"\nCalibration consistency:")
            print(f"  Mean: {mean_factor:.6f} cm²/pixel")
            print(f"  Std Dev: {std_factor:.6f}")
            print(f"  CV: {cv_percent:.2f}%")
            
            if cv_percent < 2.0:
                print("  [EXCELLENT] Consistency (<2% variation)")
            elif cv_percent < 5.0:
                print("  [GOOD] Consistency (<5% variation)")
            else:
                print("  [WARNING] Variation detected - check camera position between shots")
    
    print("\n" + "=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    test_calibration()

