#!/usr/bin/env python3
"""
Diagnostic script for hybrid OpenCV + BiRefNet quarter detection.

Shows the complete pipeline with visual outputs at each step:
1. OpenCV Hough Circle Detection
2. BiRefNet segmentation mask
3. Final overlay with measurements
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation import load_birefnet_model


def visualize_hybrid_quarter_detection(image_path: str, output_dir: str = "outputs/quarter_diagnostic"):
    """Comprehensive visualization of hybrid quarter detection pipeline."""
    
    print("=" * 80)
    print("HYBRID OPENCV + BIREFNET QUARTER DETECTION DIAGNOSTIC")
    print("=" * 80)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return False
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Extract top-left region
    roi_size = 700
    roi = image[0:roi_size, 0:roi_size].copy()
    
    # STEP 1: OpenCV Hough Circle Detection
    print("\nSTEP 1: OpenCV Hough Circle Detection")
    print("-" * 80)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=50,
        maxRadius=400
    )
    
    if circles is None:
        print("ERROR: No circles detected")
        return False
    
    # Get largest circle
    circles = np.round(circles[0, :]).astype("int")
    largest_circle = max(circles, key=lambda c: c[2])
    x, y, r = largest_circle
    
    print(f"SUCCESS: Detected circle: center=({x}, {y}), radius={r}px")
    
    # Visualize OpenCV detection
    opencv_viz = roi.copy()
    cv2.circle(opencv_viz, (x, y), r, (0, 255, 0), 3)
    cv2.circle(opencv_viz, (x, y), 2, (0, 0, 255), -1)
    cv2.putText(opencv_viz, f"OpenCV Hough", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(opencv_viz, f"Radius: {r}px", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # STEP 2: BiRefNet Segmentation
    print("\nSTEP 2: BiRefNet Segmentation")
    print("-" * 80)
    
    # Load BiRefNet model
    model, transform = load_birefnet_model()
    
    # Crop focused region around quarter
    crop_size = int(r * 3)
    crop_x1 = max(0, x - crop_size // 2)
    crop_y1 = max(0, y - crop_size // 2)
    crop_x2 = min(roi_size, x + crop_size // 2)
    crop_y2 = min(roi_size, y + crop_size // 2)
    
    quarter_crop = roi[crop_y1:crop_y2, crop_x1:crop_x2]
    print(f"Crop region: {quarter_crop.shape[1]}x{quarter_crop.shape[0]} pixels")
    
    # Convert to RGB for BiRefNet
    quarter_crop_rgb = cv2.cvtColor(quarter_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(quarter_crop_rgb)
    
    # Run BiRefNet
    input_tensor = transform(pil_image).unsqueeze(0).cuda()
    
    with torch.no_grad():
        result = model(input_tensor)
        
        if isinstance(result, list) and len(result) > 0:
            mask = result[0]
        else:
            mask = result
        
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
        
        # Apply sigmoid
        mask = 1 / (1 + np.exp(-mask))
        
        # Resize to crop size
        crop_h, crop_w = quarter_crop.shape[:2]
        if mask.shape != (crop_h, crop_w):
            mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply threshold
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            actual_area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * actual_area / (perimeter * perimeter) if perimeter > 0 else 0
            refined_radius = np.sqrt(actual_area / np.pi)
            
            print(f"SUCCESS: BiRefNet segmentation:")
            print(f"  Segmented area: {actual_area:.0f} pixels²")
            print(f"  Circularity: {circularity:.3f}")
            print(f"  Refined radius: {refined_radius:.1f}px")
            
            # Create visualization with mask overlay
            birefnet_viz = roi.copy()
            
            # Create full-size mask
            full_mask = np.zeros((roi_size, roi_size), dtype=np.uint8)
            full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = mask_binary
            
            # Apply colored overlay
            overlay = birefnet_viz.copy()
            overlay[full_mask > 0] = [0, 255, 255]  # Cyan
            birefnet_viz = cv2.addWeighted(birefnet_viz, 0.6, overlay, 0.4, 0)
            
            # Draw contours
            mask_placed = np.zeros((roi_size, roi_size), dtype=np.uint8)
            for cnt in contours:
                # Adjust contour coordinates
                cnt_adjusted = cnt + np.array([crop_x1, crop_y1])
                cv2.drawContours(mask_placed, [cnt_adjusted], -1, 255, -1)
            
            cv2.drawContours(birefnet_viz, [largest_contour + np.array([crop_x1, crop_y1])], 
                           -1, (0, 255, 255), 2)
            
            cv2.putText(birefnet_viz, f"BiRefNet Segmentation", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(birefnet_viz, f"Area: {actual_area:.0f}px", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(birefnet_viz, f"Circularity: {circularity:.3f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            print("ERROR: BiRefNet found no contours")
            birefnet_viz = roi.copy()
    
    # STEP 3: Combined visualization
    print("\nSTEP 3: Creating combined visualization")
    print("-" * 80)
    
    # Create side-by-side comparison
    combined = np.hstack([opencv_viz, birefnet_viz])
    
    # Create zoomed view of quarter
    zoom_size = int(r * 2.5)
    zoom_x1 = max(0, x - zoom_size)
    zoom_y1 = max(0, y - zoom_size)
    zoom_x2 = min(roi_size, x + zoom_size)
    zoom_y2 = min(roi_size, y + zoom_size)
    
    quarter_zoom = roi[zoom_y1:zoom_y2, zoom_x1:zoom_x2].copy()
    
    # Draw circle on zoom
    local_x = x - zoom_x1
    local_y = y - zoom_y1
    cv2.circle(quarter_zoom, (local_x, local_y), r, (0, 255, 0), 2)
    cv2.circle(quarter_zoom, (local_x, local_y), 2, (0, 0, 255), -1)
    
    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stem = Path(image_path).stem
    
    cv2.imwrite(str(output_path / f"1_opencv_{stem}.jpg"), opencv_viz)
    cv2.imwrite(str(output_path / f"2_birefnet_{stem}.jpg"), birefnet_viz)
    cv2.imwrite(str(output_path / f"3_combined_{stem}.jpg"), combined)
    cv2.imwrite(str(output_path / f"4_zoom_{stem}.jpg"), quarter_zoom)
    
    print(f"\nSUCCESS: Saved diagnostic visualizations to: {output_path}/")
    print(f"  1_opencv_{stem}.jpg - OpenCV circle detection")
    print(f"  2_birefnet_{stem}.jpg - BiRefNet segmentation")
    print(f"  3_combined_{stem}.jpg - Side-by-side comparison")
    print(f"  4_zoom_{stem}.jpg - Zoomed quarter view")
    
    # Calculate calibration
    QUARTER_AREA_CM2 = 4.62
    calibration_factor = QUARTER_AREA_CM2 / actual_area
    
    print("\nCALIBRATION RESULTS:")
    print("-" * 80)
    print(f"Quarter area (actual): {QUARTER_AREA_CM2} cm²")
    print(f"Quarter area (pixels): {actual_area:.0f} px²")
    print(f"Calibration factor: {calibration_factor:.6f} cm²/pixel")
    print(f"Equivalent: {1/calibration_factor:.2f} pixels per cm²")
    
    return True


def main():
    """Main diagnostic function."""
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "test_images/IMG_8787.JPG"
    
    print(f"\nTest image: {test_image}\n")
    
    success = visualize_hybrid_quarter_detection(test_image)
    
    print("\n" + "=" * 80)
    if success:
        print("DIAGNOSTIC COMPLETE - Check outputs/quarter_diagnostic/ for visualizations")
    else:
        print("DIAGNOSTIC FAILED - See error messages above")
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

