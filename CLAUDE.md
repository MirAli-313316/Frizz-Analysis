# CRITICAL PROJECT RULES - ALWAYS FOLLOW

## File References
- ALWAYS refer to context/project_spec.md for detailed specifications
- ALWAYS refer to context/time_detection_rules.md for filename parsing logic  
- ALWAYS refer to context/features/surface_area_calculation.md for measurement formulas
- Test images are in test_images/ directory
- All outputs should go to outputs/ directory (create if not exists)

## Development Approach
- Implement incrementally - get basic functionality working first
- Test with actual images in test_images/ after each feature
- Use type hints for all functions
- Add docstrings explaining the science/purpose
- Log important steps for debugging (calibration values, detection counts)

## Important Project Context
- ALWAYS refer to context/design_decisions.md for architecture choices
- ALWAYS refer to context/known_issues.md before suggesting changes
- Image processing uses resize strategy BY DESIGN (not a limitation)

## Error Handling
- Never fail silently - always log or display issues
- Provide helpful error messages that suggest solutions
- Continue processing other images even if one fails

## Testing
- Every new function should be testable independently
- Create simple test scripts that visualize results
- Show intermediate steps (detected quarter, segmented tresses) for verification

# Frizz Test Analysis Application

This is a scientific image analysis application for quantitative frizz testing of hair tresses using computer vision and segmentation models.

## Application Purpose
Analyze time-series images of hair tresses in humidity-controlled conditions to quantitatively measure anti-frizz product efficacy by calculating surface area changes over time.

## Technical Specifications

### Image Properties
- Camera: Canon JPEG, 5184x3456 pixels (18MP)
- File naming: Default (IMG_XXXX) or custom (0-hour, 1-hour, etc.)
- Background: White backdrop
- Calibration: US quarter (24.26mm diameter) in top-left corner
- Setup: Can handle 1 to 10+ tresses

### Time Point Detection Logic
- If default names (IMG_XXXX): Sort by number, assign time points in order
  - Standard sequence: 0h, 30min, 1h, 2h, 4h, 6h, 8h, 24h
- If renamed: Parse time from filename (e.g., "0-hour.jpg", "30-min.jpg")
- Flexible: Handle any number of images (minimum 2 for comparison)

### Measurement Requirements
- Units: cm²
- Per-image calibration using quarter (mandatory)
- Works with single tress or multiple
- Calculate total surface area per tress
- Track percentage change from 0-hour baseline

## Code Architecture

### Core Modules
- `calibration.py` - Quarter detection and pixel-to-cm² conversion (per image)
- `segmentation.py` - Hair tress detection using SAM 2
- `analysis.py` - Surface area calculations
- `time_parser.py` - Intelligent time point detection from filenames
- `batch_processor.py` - Time-series processing
- `gui.py` - Tkinter desktop interface
- `visualization.py` - Overlay generation and charts
- `excel_reporter.py` - Multi-sheet Excel generation

### Hardware Configuration
- GPU: NVIDIA GeForce RTX 3050 Ti (4GB VRAM)
- Automatically use CUDA when available
- Fallback to CPU if needed (with warning)

## Development Guidelines

### Image Processing Pipeline
1. Load image and detect quarter (calibrate individually)
2. Calculate calibration ratio for THIS image
3. Run SAM 2 on full image
4. Identify individual tresses (works with 1+)
5. Calculate surface areas using calibration
6. Store results with time point metadata

### Time Point Handling
```python
# Priority order for time detection:
1. Check if filename contains time markers (0-hour, 1h, 30-min, etc.)
2. If numeric only (IMG_8780), sort and map to standard sequence
3. Allow user override in GUI