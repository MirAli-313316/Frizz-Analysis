# Surface Area Calculation Specification

## Calibration Method
Each image MUST be calibrated independently using its own quarter detection.

### Quarter Reference
- Actual diameter: 24.26 mm = 2.426 cm
- Actual area: π × (1.213)² = 4.62 cm²

### Calibration Formula
```python
# For each image:
quarter_pixels = detect_quarter_pixel_area(image)
calibration_factor = 4.62 / quarter_pixels  # cm²/pixel