# Design Decisions & Technical Solutions

## Image Resolution Strategy
**Decision**: Use 18MP (5184×3456) capture with software resizing for processing
**Date**: Initial design
**Reason**: 
- Maximum detail capture for scientific accuracy (fine frizz patterns)
- Hair frizz analysis requires high-resolution edge detection
- Storage is not a constraint vs. accuracy loss
**Implementation**:
- Capture at full 18MP
- Resize to 1024px max dimension for SAM processing
- Scale masks back to original resolution for measurement
- Preserves source images for validation/publication

## Memory Management for SAM
**Challenge**: RTX 3050 Ti (4GB VRAM) cannot process 18MP images directly
- Uncompressed 18MP image = 215 MB
- SAM processing needs ~12.8 GB for full resolution
**Solution**: Two-stage processing pipeline
1. Resize image to 1024px for mask generation (50× less memory)
2. Scale binary masks back to original resolution
3. Perform measurements on full-resolution masks
**Rationale**: Industry-standard approach, maintains accuracy while enabling GPU acceleration
**Parameters**:
- SAM points_per_side: 16 (reduced from 32)
- SAM crop_n_layers: 0 (disabled multi-scale)
- min_mask_region_area: 500 (for resized images)

## Calibration Strategy  
**Decision**: Calibrate each image independently
**Reason**: 
- Accounts for minor camera movement between shots
- Ensures maximum accuracy across time series
- Scientific best practice for quantitative imaging
**Implementation**: Quarter detection runs on every image, stores calibration factor per image

## Time Point Detection
**Decision**: Dual approach - filename parsing with fallback
**Implementation**:
1. First attempt: Parse explicit time markers from filename
2. Fallback: Use position in sorted batch
**Reason**: Flexibility for different user workflows while maintaining automation

## Frame Detection Strategy
**Decision**: Shape-based filtering instead of color-based
**Date**: 2025-10-02
**Challenge**: Rectangular frames/boxes being detected as tresses
**Solution**: Multi-criteria frame detection
1. **Rectangularity threshold (> 0.85)**: Filters objects that perfectly fill their bounding box
2. **Narrow strip detection (aspect ratio > 6.5)**: Catches frame edges
3. **Solidity check (< 0.15)**: Detects hollow frame structures
4. **Edge-touching detection**: Filters large objects spanning multiple edges
**Rationale**: 
- Shape-based approach works regardless of frame color (cyan, black, white, etc.)
- Allows tresses of any shape/morphology (frizzy, narrow, cloud-like)
- No aspect ratio constraints on actual tresses
**Implementation**: See `_is_frame_or_background()` in segmentation.py
**Threshold Values**:
- rectangularity_threshold: 0.85
- narrow_strip_aspect_ratio: 6.5
- solidity_threshold: 0.15

## Merged Tress Splitting
**Decision**: Automatic splitting of large masks using morphological operations
**Date**: 2025-10-02
**Challenge**: SAM sometimes detects multiple tresses as single merged object
**Solution**: Post-processing split pipeline
1. **Detection trigger**: Masks > 850,000 pixels (~50 cm²)
2. **Morphological opening**: Breaks weak connections (15×15 kernel)
3. **Erosion**: Further separates touching regions (5×5 kernel, 2 iterations)
4. **Connected components**: Identifies separate objects
5. **Dilation recovery**: Restores original boundaries (7×7 kernel)
6. **Boundary constraint**: Intersects with original mask
**Rationale**:
- Provides individual tress measurements instead of merged totals
- Better granularity for tracking frizz changes per tress
- Conservative approach with safety checks
- Only activates for suspiciously large masks
**Implementation**: See `_split_merged_tresses()` in segmentation.py
**Parameters**:
- split_threshold: 850,000 pixels (~50 cm²)
- min_component_area: 200,000 pixels (~10 cm²)
- opening_kernel: 15×15 ellipse
- erosion_kernel: 5×5 ellipse (2 iterations)
- dilation_kernel: 7×7 ellipse (2 iterations)

## Segmentation Diagnostic Logging
**Decision**: Comprehensive logging for all mask filtering decisions
**Date**: 2025-10-02
**Reason**: 
- Transparency in segmentation decisions
- Easy debugging when results unexpected
- Scientific reproducibility and validation
**Implementation**: Logs for each mask include:
- Area, bounding box, position
- Rectangularity, solidity, brightness
- Edges touched, color analysis
- Accept/reject decision with specific reason
**Format**: Structured logs with clear visual indicators (✓, ❌, ⚠️)