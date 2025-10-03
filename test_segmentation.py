"""
Test script for segmentation module.

This script tests hair tress detection using SAM on images in test_images/ directory
and displays visualizations with colored overlays.
"""

import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from calibration import detect_quarter
from segmentation import segment_tresses, create_visualization, create_exclude_region_from_quarter


def test_segmentation():
    """Test segmentation on all images in test_images directory."""
    
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
    print("HAIR TRESS SEGMENTATION TEST")
    print("=" * 80)
    
    results = []
    
    for image_path in sorted(image_files):
        print(f"\n{image_path.name}")
        print("-" * 80)
        
        try:
            # Step 1: Detect quarter for exclusion region
            print("Step 1: Detecting quarter for exclusion region...")
            calibration = detect_quarter(str(image_path))
            
            # Create exclusion region around quarter
            exclude_region = create_exclude_region_from_quarter(
                calibration.quarter_center,
                calibration.quarter_radius,
                margin=50
            )
            print(f"  Quarter at {calibration.quarter_center}, radius {calibration.quarter_radius}px")
            print(f"  Exclusion region: {exclude_region}")
            
            # Step 2: Segment tresses
            print("\nStep 2: Segmenting hair tresses with SAM...")
            segmentation = segment_tresses(
                str(image_path),
                exclude_region=exclude_region
            )
            
            print(f"\n[OK] Segmentation successful!")
            print(f"  Tresses detected: {len(segmentation.tresses)}")
            print(f"  Device used: {segmentation.device_used.upper()}")
            print(f"  Processing time: {segmentation.processing_time:.2f}s")
            
            # Print details for each tress
            if segmentation.tresses:
                print(f"\n  Tress Details:")
                for tress in segmentation.tresses:
                    x, y, w, h = tress.bbox
                    print(f"    Tress {tress.id}:")
                    print(f"      Area: {tress.area_pixels:.0f} pixels")
                    print(f"      Position: ({x}, {y})")
                    print(f"      Size: {w}x{h} pixels")
                    print(f"      Aspect ratio: {tress.aspect_ratio:.2f}")
                    print(f"      Confidence: {tress.confidence:.2f}")
            else:
                print("\n  [WARNING] No tresses detected!")
                print("  Possible reasons:")
                print("    - Tresses may be too small (adjust min_tress_area)")
                print("    - Aspect ratio filter may be too strict")
                print("    - Brightness threshold may need adjustment")
            
            # Step 3: Create visualization
            print(f"\nStep 3: Creating visualization...")
            vis_image = create_visualization(str(image_path), segmentation)
            
            # Save visualization
            output_path = output_dir / f"segmentation_{image_path.stem}.jpg"
            cv2.imwrite(str(output_path), vis_image)
            print(f"  Saved: {output_path}")
            
            # Display visualization
            # Resize for display if too large
            display_image = vis_image.copy()
            max_display_width = 1920
            height, width = display_image.shape[:2]
            if width > max_display_width:
                scale = max_display_width / width
                new_width = max_display_width
                new_height = int(height * scale)
                display_image = cv2.resize(display_image, (new_width, new_height))
            
            cv2.imshow(f"Segmentation: {image_path.name}", display_image)
            print(f"\nVisualization displayed. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            results.append({
                'filename': image_path.name,
                'success': True,
                'num_tresses': len(segmentation.tresses),
                'tresses': segmentation.tresses,
                'device': segmentation.device_used,
                'time': segmentation.processing_time
            })
            
        except Exception as e:
            print(f"[ERROR] Segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'filename': image_path.name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\nTotal images: {total}")
    print(f"Successful segmentations: {successful}")
    print(f"Failed segmentations: {total - successful}")
    
    if successful > 0:
        print("\nDetection results:")
        total_tresses = 0
        total_time = 0.0
        
        for result in results:
            if result['success']:
                num = result['num_tresses']
                device = result['device'].upper()
                time = result['time']
                total_tresses += num
                total_time += time
                
                print(f"  {result['filename']}: {num} tresses ({device}, {time:.2f}s)")
        
        print(f"\nTotal tresses across all images: {total_tresses}")
        print(f"Average tresses per image: {total_tresses / successful:.1f}")
        print(f"Average processing time: {total_time / successful:.2f}s")
        
        # Show pixel areas
        print("\nPixel areas by image:")
        for result in results:
            if result['success'] and result['tresses']:
                print(f"\n  {result['filename']}:")
                for tress in result['tresses']:
                    print(f"    Tress {tress.id}: {tress.area_pixels:.0f} pixels")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("\nVisualization images saved to outputs/ directory")
    print("Each tress is shown in a different color with labels")


if __name__ == "__main__":
    test_segmentation()

