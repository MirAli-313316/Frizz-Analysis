#!/usr/bin/env python3
"""
Simple test script for BiRefNet integration.

Tests the basic functionality of the BiRefNet hair frizz analysis pipeline.

Usage:
    python test_birefnet_integration.py [image_path]

If no image path is provided, uses test_images/IMG_8787.JPG
"""

import sys
import time
import traceback
from pathlib import Path

try:
    import cv2
    import numpy as np

    # Import our modules (absolute imports)
    from src.analysis import analyze_image
    from src.segmentation import load_birefnet_model, segment_all_tresses
    from src.tress_detector import detect_tress_regions

except ImportError as e:
    print(f"ERROR: Missing required modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def test_model_loading():
    """Test BiRefNet model loading and caching."""
    print("=" * 60)
    print("TESTING BIREFNET MODEL LOADING")
    print("=" * 60)

    try:
        model, transform = load_birefnet_model()
        print("SUCCESS: BiRefNet model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Transform type: {type(transform).__name__}")

        # Test second load (should use cache)
        start_time = time.time()
        model2, transform2 = load_birefnet_model()
        cache_time = time.time() - start_time

        print(f"SUCCESS: Model caching works (loaded in {cache_time:.4f}s)")
        print(f"   Same objects returned: {model is model2 and transform is transform2}")

        return True

    except Exception as e:
        print(f"FAILED: Model loading failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False


def test_tress_detection(image_path: str):
    """Test OpenCV tress detection."""
    print("=" * 60)
    print("TESTING TRESS DETECTION")
    print("=" * 60)

    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"FAILED: Could not load image: {image_path}")
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"SUCCESS: Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

        # Detect tresses
        start_time = time.time()
        tress_boxes = detect_tress_regions(image_rgb, num_expected_tresses=7)
        detection_time = time.time() - start_time

        print(f"SUCCESS: Tress detection completed in {detection_time:.2f}s")
        print(f"   Found {len(tress_boxes)} tress regions")

        if tress_boxes:
            print("   Top 3 tresses by size:")
            # Sort by area for display
            tress_areas = [(i, w * h, (x, y, w, h))
                          for i, (x, y, w, h) in enumerate(tress_boxes, 1)]
            tress_areas.sort(key=lambda x: x[1], reverse=True)

            for i, (tress_id, area, bbox) in enumerate(tress_areas[:3], 1):
                x, y, w, h = bbox
                print(f"     Tress {tress_id}: {area:,} pixels ({w}x{h}) at ({x}, {y})")

        return len(tress_boxes) > 0

    except Exception as e:
        print(f"FAILED: Tress detection failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False


def test_segmentation(image_path: str, max_tresses: int = 3):
    """Test BiRefNet segmentation on detected tresses."""
    print("=" * 60)
    print("TESTING BIREFNET SEGMENTATION")
    print("=" * 60)

    try:
        # Load image and detect tresses
        image = cv2.imread(image_path)
        if image is None:
            print(f"FAILED: Could not load image: {image_path}")
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tress_boxes = detect_tress_regions(image_rgb, num_expected_tresses=7)

        if not tress_boxes:
            print("FAILED: No tresses detected for segmentation testing")
            return False

        print(f"SUCCESS: Found {len(tress_boxes)} tresses for segmentation")
        print(f"   Testing segmentation on first {min(max_tresses, len(tress_boxes))} tresses")

        # Load BiRefNet model
        model, transform = load_birefnet_model()

        # Test segmentation on subset of tresses
        test_boxes = tress_boxes[:max_tresses]

        start_time = time.time()
        seg_result = segment_all_tresses(image_rgb, test_boxes, model, transform)
        segmentation_time = time.time() - start_time

        print(f"SUCCESS: Segmentation completed in {segmentation_time:.2f}s")
        print(f"   Processed {len(seg_result.tresses)} tresses")
        print(f"   Average time per tress: {segmentation_time/len(seg_result.tresses):.2f}s")

        # Show results for each tress
        print("   Segmentation results:")
        for tress in seg_result.tresses:
            area_pixels = tress.area_pixels
            confidence = tress.confidence
            bbox = tress.bbox
            print(f"     Tress {tress.id}: {area_pixels:,} pixels, conf: {confidence:.3f}")

        return True

    except Exception as e:
        print(f"FAILED: Segmentation test failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False


def test_full_pipeline(image_path: str):
    """Test complete analysis pipeline."""
    print("=" * 60)
    print("TESTING COMPLETE ANALYSIS PIPELINE")
    print("=" * 60)

    try:
        start_time = time.time()

        # Run complete analysis
        result = analyze_image(
            image_path,
            visualize=True,
            output_dir="outputs/test_birefnet",
            num_expected_tresses=6
        )

        total_time = time.time() - start_time

        print(f"SUCCESS: Complete analysis completed in {total_time:.2f}s")
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

        return True

    except Exception as e:
        print(f"FAILED: Complete pipeline test failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False


def test_visualization(image_path: str):
    """Test visualization generation."""
    print("=" * 60)
    print("TESTING VISUALIZATION GENERATION")
    print("=" * 60)

    try:
        # Load image and get segmentation results
        image = cv2.imread(image_path)
        if image is None:
            print(f"FAILED: Could not load image: {image_path}")
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get segmentation results
        model, transform = load_birefnet_model()
        tress_boxes = detect_tress_regions(image_rgb, num_expected_tresses=7)[:2]  # Test with 2 tresses
        seg_result = segment_all_tresses(image_rgb, tress_boxes, model, transform)

        # Generate visualization
        from src.segmentation import visualize_segmentation
        viz_image = visualize_segmentation(image_rgb, seg_result)

        # Save visualization
        output_path = "outputs/test_birefnet/visualization_test.jpg"
        success = cv2.imwrite(output_path, cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))

        if success:
            print(f"SUCCESS: Visualization saved to: {output_path}")
            print(f"   Image size: {viz_image.shape[1]}x{viz_image.shape[0]} pixels")
            return True
        else:
            print("FAILED: Failed to save visualization")
            return False

    except Exception as e:
        print(f"FAILED: Visualization test failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main test function."""
    print("BIREFNET INTEGRATION TEST SUITE")
    print("=" * 60)
    print("Testing the complete BiRefNet hair frizz analysis pipeline")
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

    # Run all tests
    tests = [
        ("Model Loading", test_model_loading),
        ("Tress Detection", lambda: test_tress_detection(test_image)),
        ("BiRefNet Segmentation", lambda: test_segmentation(test_image)),
        ("Complete Pipeline", lambda: test_full_pipeline(test_image)),
        ("Visualization", lambda: test_visualization(test_image)),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"FAILED: {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! BiRefNet integration is working correctly.")
        print("   You can now run the full application with: python run_gui.py")
        return 0
    else:
        print(f"\nWARNING: {total - passed} test(s) failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
