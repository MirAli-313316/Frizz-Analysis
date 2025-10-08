"""
Diagnostic script to understand why tress detection is failing.
"""

import cv2
import numpy as np
from pathlib import Path

# Load test image
test_image = "test_images/IMG_8787.JPG"
image = cv2.imread(test_image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Image shape: {image_rgb.shape}")
print(f"Image dtype: {image_rgb.dtype}")

# Convert to grayscale
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Analyze pixel values
print(f"\nGrayscale statistics:")
print(f"  Min: {gray.min()}")
print(f"  Max: {gray.max()}")
print(f"  Mean: {gray.mean():.1f}")
print(f"  Median: {np.median(gray):.1f}")

# Test different thresholds
thresholds = [150, 180, 200, 220, 240]
print(f"\nBinary threshold analysis (THRESH_BINARY_INV):")
print(f"  Threshold | Pixels > threshold (dark hair) | Percentage")
print(f"  ----------|-------------------------------|------------")

for thresh in thresholds:
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    white_pixels = np.sum(binary == 255)
    percentage = (white_pixels / binary.size) * 100
    print(f"  {thresh:>9} | {white_pixels:>29} | {percentage:>10.2f}%")

# Try threshold 200 (current setting)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\nContours found: {len(contours)}")
print(f"\nTop 10 contours by area:")
print(f"  Rank | Area (pixels) | Bounding Box (x, y, w, h)")
print(f"  -----|---------------|---------------------------")

# Sort by area
contour_areas = [(i, cv2.contourArea(c), cv2.boundingRect(c)) for i, c in enumerate(contours)]
contour_areas.sort(key=lambda x: x[1], reverse=True)

for rank, (idx, area, bbox) in enumerate(contour_areas[:10], 1):
    x, y, w, h = bbox
    print(f"  {rank:>4} | {area:>13.0f} | ({x}, {y}, {w}, {h})")

# Check if the largest contour is the entire image
if contour_areas:
    _, largest_area, largest_bbox = contour_areas[0]
    x, y, w, h = largest_bbox
    if w == image_rgb.shape[1] and h == image_rgb.shape[0]:
        print(f"\n⚠️  WARNING: Largest contour is the ENTIRE IMAGE!")
        print(f"   This means the binary threshold detected almost everything as 'hair'")
        print(f"   The image might have a darker background than expected")
    
    # Filter by minimum area (50000)
    filtered = [c for c in contour_areas if c[1] >= 50000]
    print(f"\nContours with area >= 50,000 pixels: {len(filtered)}")

# Save diagnostic visualizations
output_dir = Path("outputs/diagnostics")
output_dir.mkdir(parents=True, exist_ok=True)

# Save binary threshold result
cv2.imwrite(str(output_dir / "binary_threshold.jpg"), binary)
print(f"\n✓ Saved binary threshold visualization: outputs/diagnostics/binary_threshold.jpg")

# Save contour visualization
contour_viz = image.copy()
cv2.drawContours(contour_viz, contours, -1, (0, 255, 0), 3)
cv2.imwrite(str(output_dir / "contours.jpg"), contour_viz)
print(f"✓ Saved contour visualization: outputs/diagnostics/contours.jpg")

# Recommendation
print(f"\n{'='*70}")
print(f"RECOMMENDATIONS:")
print(f"{'='*70}")

if contour_areas and contour_areas[0][1] > (image_rgb.shape[0] * image_rgb.shape[1] * 0.9):
    print("• The threshold is detecting almost the entire image as 'hair'")
    print("• This suggests the image might have:")
    print("  - A darker/gray background instead of white")
    print("  - Overall darker lighting")
    print("• Try INCREASING the threshold (e.g., 220-240)")
    print("• Or check if this test image is appropriate for the detection algorithm")
elif len(filtered) > 1:
    print(f"• Detection found {len(filtered)} potential tresses")
    print("• This looks promising! Check the visualizations.")
else:
    print("• Only 1 large region detected")
    print("• Image might not have separate tresses, or they're too close together")
    print("• Check outputs/diagnostics/ for visual confirmation")
