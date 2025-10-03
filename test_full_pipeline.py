"""
Test the complete analysis pipeline: calibration + segmentation + reporting.

This script processes all images in test_images/ and generates:
- Individual visualizations showing detected tresses and areas
- Comprehensive Excel report with multiple sheets
- Summary statistics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.batch_processor import process_directory
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete pipeline test."""
    
    print("\n" + "=" * 70)
    print("FRIZZ ANALYSIS - FULL PIPELINE TEST")
    print("=" * 70)
    print("\nThis test will:")
    print("  1. Detect quarters and calibrate each image")
    print("  2. Segment hair tresses using SAM")
    print("  3. Calculate surface areas in cm²")
    print("  4. Track changes over time")
    print("  5. Generate Excel report with multiple sheets")
    print("  6. Create visualizations for each image")
    print("\n" + "=" * 70)
    
    # Configuration
    test_images_dir = "test_images"
    output_dir = "outputs"
    pattern = "*.JPG"  # Adjust if needed (e.g., "*.jpg", "IMG_*.JPG")
    
    # Check if test_images directory exists
    if not Path(test_images_dir).exists():
        logger.error(f"Directory not found: {test_images_dir}")
        logger.error("Please create test_images/ and add your test images")
        return
    
    # Check if any images exist
    image_files = list(Path(test_images_dir).glob(pattern))
    if not image_files:
        # Try lowercase
        pattern = "*.jpg"
        image_files = list(Path(test_images_dir).glob(pattern))
    
    if not image_files:
        logger.error(f"No images found in {test_images_dir}")
        logger.error(f"Supported formats: *.JPG, *.jpg")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    try:
        # Process all images
        results, excel_path = process_directory(
            directory=test_images_dir,
            pattern=pattern,
            output_dir=output_dir,
            visualize=True,
            excel_filename="test_results.xlsx",
            max_processing_dim=1024  # Memory-optimized for RTX 3050 Ti
        )
        
        # Display summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        for result in results:
            print(f"\n{result.image_name}:")
            print(f"  Tresses detected: {len(result.tresses)}")
            print(f"  Total area: {result.get_total_area():.2f} cm²")
            print(f"  Individual tress areas:")
            for tress in result.tresses:
                print(f"    Tress {tress.tress_id}: {tress.area_cm2:.2f} cm² "
                      f"({tress.pixel_count:,} pixels)")
        
        print("\n" + "=" * 70)
        print("OUTPUT FILES")
        print("=" * 70)
        print(f"\n[OK] Excel Report: {excel_path}")
        print(f"[OK] Visualizations: {output_dir}/analysis_*.jpg")
        print(f"[OK] Calibration checks: {output_dir}/calibration_*.jpg")
        
        print("\n" + "=" * 70)
        print("QUALITY CHECKS")
        print("=" * 70)
        
        # Verify expected surface area ranges
        all_areas = []
        for result in results:
            for tress in result.tresses:
                all_areas.append(tress.area_cm2)
        
        if all_areas:
            min_area = min(all_areas)
            max_area = max(all_areas)
            avg_area = sum(all_areas) / len(all_areas)
            
            print(f"\nSurface Area Statistics:")
            print(f"  Min: {min_area:.2f} cm²")
            print(f"  Max: {max_area:.2f} cm²")
            print(f"  Average: {avg_area:.2f} cm²")
            print(f"  Total tresses: {len(all_areas)}")
            
            # Check if areas are in expected range (5-30 cm² per tress)
            if min_area < 5:
                print(f"\n[WARNING] Some tresses are smaller than expected (< 5 cm²)")
                print(f"  This might indicate incorrect calibration or small debris")
            
            if max_area > 30:
                print(f"\n[WARNING] Some tresses are larger than expected (> 30 cm²)")
                print(f"  This might indicate merged tresses or incorrect segmentation")
            
            if 5 <= min_area and max_area <= 30:
                print(f"\n[OK] All surface areas are in expected range (5-30 cm²)")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Open the Excel report to view detailed results:")
        print(f"   {excel_path}")
        print("\n2. Check visualizations to verify tress detection:")
        print(f"   {output_dir}/analysis_*.jpg")
        print("\n3. Review calibration images for quarter detection:")
        print(f"   {output_dir}/calibration_*.jpg")
        print("\n4. If results look good, process your full dataset!")
        
        print("\n" + "=" * 70)
        print("TEST COMPLETE [SUCCESS]")
        print("=" * 70 + "\n")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}", exc_info=True)
        print("\n" + "=" * 70)
        print("TEST FAILED [ERROR]")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure all images have a visible quarter in the top-left")
        print("  - Check that SAM model is downloaded (models/sam_vit_b.pth)")
        print("  - Verify CUDA is working (or CPU fallback is enabled)")
        print("  - Check log messages above for specific errors")
        print("\n" + "=" * 70 + "\n")
        return


if __name__ == "__main__":
    main()

