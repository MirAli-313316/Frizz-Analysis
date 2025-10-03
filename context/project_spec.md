# Frizz Test Analysis - Detailed Specifications

## Image Analysis Pipeline

### Stage 1: Calibration
- Detect US quarter (24.26mm diameter)
- Location: Top-left corner
- Mounting: Translucent plastic backing
- Use Hough circles with constraints:
  - Expected size range based on distance
  - Color: Metallic gray/silver
  - Circular shape validation

### Stage 2: Tress Detection
- Expected count: 5-10 tresses
- Arrangement: 2 horizontal rows
- Characteristics:
  - Dark against white background
  - Vertical orientation (hung)
  - May have irregular edges (frizz)
  - Width varies with frizz level

### Stage 3: Measurement
- Surface area = pixel_count * calibration_factor
- Calibration: quarter_area_cm2 / quarter_area_pixels
- Each tress measured independently
- Store coordinates for visualization

### Stage 4: Analysis
- Calculate % change: ((area_t - area_0) / area_0) * 100
- Time points: 0, 1, 2, 4, 8, 16, 24 hours
- Statistical: mean, std dev across tresses

## Expected Challenges
1. Quarter on translucent plastic - may affect edge detection
2. Overlapping tresses - need separation algorithm
3. Varying frizz patterns - affects segmentation
4. Shadow variations despite controlled lighting

## Output Requirements
- Excel: One sheet per time point, summary sheet with % changes
- Visualizations: Overlay showing detected regions with labels
- Graphs: Surface area vs time for each tress
- Statistics: Average frizz increase over time