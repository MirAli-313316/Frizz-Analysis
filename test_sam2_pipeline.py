"""
Test script for SAM 2 crop-based segmentation pipeline.

Tests the complete workflow:
1. Model download check
2. Tress detection
3. SAM 2 segmentation
4. Analysis pipeline
"""

import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_availability():
    """Check if SAM 2 model is available."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: SAM 2 Model Availability")
    logger.info("="*70)
    
    model_path = Path("models/sam2_hiera_large.pt")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Model found: {model_path}")
        logger.info(f"✓ Size: {size_mb:.1f} MB")
        return True
    else:
        logger.warning(f"✗ Model not found: {model_path}")
        logger.warning("  Run: python download_sam2_model.py")
        return False


def test_tress_detection():
    """Test tress detection on a sample image."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Tress Detection (OpenCV)")
    logger.info("="*70)
    
    try:
        import cv2
        import numpy as np
        from src.tress_detector import detect_tress_regions, visualize_tress_detection
        
        # Find a test image
        test_images = list(Path("test_images").glob("*.JPG")) + list(Path("test_images").glob("*.jpg"))
        
        if not test_images:
            logger.warning("✗ No test images found in test_images/")
            return False
        
        test_image = str(test_images[0])
        logger.info(f"Testing with: {Path(test_image).name}")
        
        # Load image
        image = cv2.imread(test_image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect tresses (without calibration box for this test)
        tress_boxes = detect_tress_regions(image_rgb, calibration_box=None)
        
        if len(tress_boxes) > 0:
            logger.info(f"✓ Detected {len(tress_boxes)} tresses")
            for i, (x, y, w, h) in enumerate(tress_boxes, 1):
                logger.info(f"  Tress {i}: {w}x{h} pixels at ({x}, {y})")
            
            # Create visualization
            viz = visualize_tress_detection(image_rgb, tress_boxes)
            output_path = Path("outputs/test_tress_detection.jpg")
            output_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(output_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
            logger.info(f"✓ Saved visualization: {output_path}")
            
            return True
        else:
            logger.warning("✗ No tresses detected")
            return False
            
    except Exception as e:
        logger.error(f"✗ Tress detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sam2_loading():
    """Test SAM 2 model loading."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: SAM 2 Model Loading")
    logger.info("="*70)
    
    try:
        from src.segmentation import load_sam2_model, get_device
        
        # Check device
        device = get_device()
        logger.info(f"Device: {device}")
        
        # Load model
        logger.info("Loading SAM 2 model...")
        predictor = load_sam2_model()
        
        logger.info("✓ SAM 2 model loaded successfully")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        logger.error("  Run: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        return False
    except FileNotFoundError as e:
        logger.error(f"✗ Model file not found")
        logger.error("  Run: python download_sam2_model.py")
        return False
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_analysis():
    """Test full analysis pipeline on a sample image."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Full Analysis Pipeline")
    logger.info("="*70)
    
    try:
        from src.analysis import analyze_image
        
        # Find a test image
        test_images = list(Path("test_images").glob("*.JPG")) + list(Path("test_images").glob("*.jpg"))
        
        if not test_images:
            logger.warning("✗ No test images found in test_images/")
            return False
        
        test_image = str(test_images[0])
        logger.info(f"Testing with: {Path(test_image).name}")
        
        # Run full analysis
        result = analyze_image(
            test_image,
            visualize=True,
            output_dir="outputs/test",
            num_expected_tresses=7
        )
        
        logger.info(f"✓ Analysis complete!")
        logger.info(f"  Tresses detected: {len(result.tresses)}")
        logger.info(f"  Total area: {result.get_total_area():.2f} cm²")
        logger.info(f"  Processing time: {result.processing_time:.2f}s")
        logger.info(f"  Device: {result.device_used}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Full analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("SAM 2 CROP-BASED SEGMENTATION PIPELINE TEST")
    logger.info("="*70)
    
    results = {
        "Model Availability": test_model_availability(),
        "Tress Detection": test_tress_detection(),
        "SAM 2 Loading": test_sam2_loading(),
        "Full Analysis": test_full_analysis()
    }
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:.<50} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    logger.info("="*70)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("="*70)
    
    if passed == total:
        logger.info("\n✓ All tests passed! Pipeline is ready to use.")
        return 0
    else:
        logger.warning(f"\n✗ {total - passed} test(s) failed. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
