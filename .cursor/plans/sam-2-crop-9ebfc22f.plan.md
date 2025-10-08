<!-- 9ebfc22f-f5e6-4e6e-8ba7-9a21f83938e5 4e962575-2a4b-4ac8-8d5b-b32c9f66dc78 -->
# Optimize Segmentation Pipeline with SAM 2 Crop-Based Processing

## Phase 1: Environment Setup

### Update dependencies

- Update `requirements.txt`: Remove `segment-anything`, add SAM 2 package
- Install command: `pip install git+https://github.com/facebookresearch/segment-anything-2.git`

### Add SAM 2 model download functionality

- Create `download_sam2_model.py` similar to existing `download_sam_model.py`
- Auto-download `sam2_hiera_large.pt` from Meta's official repository
- Check if `models/sam2_hiera_large.pt` exists, download if missing
- Include progress bar during download

## Phase 2: Tress Detection Module

### Create `src/tress_detector.py`

New module using OpenCV for fast, lightweight tress detection:

**Core detection function**: `detect_tress_regions(image, calibration_box, num_expected_tresses=7)`

- Convert to grayscale
- Binary threshold at 200 (hair dark, background white)
- Use `cv2.THRESH_BINARY_INV` for white-on-black
- Mask out calibration quarter region
- Morphological closing with 15x15 kernel to connect strands
- Find external contours
- Filter by minimum area (50,000+ pixels)
- Get bounding rectangles with 100px padding
- Sort by position (y, then x) for consistent ordering
- Return list of (x, y, w, h) bounding boxes

**Visualization function**: `visualize_tress_detection(image, boxes)`

- Draw green rectangles around detected tresses
- Label each as "Tress 1", "Tress 2", etc.
- Return annotated image for saving

## Phase 3: Segmentation Module Refactor

### Update `src/segmentation.py`

**Replace SAM 1 with SAM 2**:

- Remove: `from segment_anything import sam_model_registry, SamAutomaticMaskGenerator`
- Add: `from sam2.sam2_image_predictor import SAM2ImagePredictor`
- Update `load_sam_model()` to load SAM 2 predictor (not automatic mask generator)
- Model path: `models/sam2_hiera_large.pt`
- Auto-download if missing

**New function**: `segment_single_tress(tress_crop, crop_box, full_image_shape, predictor)`

- Input: Cropped image region containing one tress
- Smart resizing:
- If max(h,w) > 2500px: resize to 2048px max using INTER_AREA
- If max(h,w) ≤ 2500px: process at native resolution
- Track scale_factor for mask rescaling
- Generate point prompts:
- Create 48x48 grid across crop
- Filter: keep only points where pixel mean < 150 (dark/hair)
- Fallback: use center point if < 5 points found
- Convert to numpy arrays: input_points, input_labels (all 1s)
- Call SAM 2:
- `predictor.set_image(resized_crop)`
- `predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)`
- Scale mask back to original crop size if resized (use INTER_NEAREST)
- Stitch into full image coordinates: place at (x, y) position
- Return TressMask with crop and full-image masks

**New function**: `segment_all_tresses(image, tress_boxes, calibration_box, predictor)`

- Loop through each bounding box
- Extract crop: `crop = image[y:y+h, x:x+w]`
- Call `segment_single_tress()` for each
- Log progress: "Processing tress X/N: WxH px crop"
- Clear GPU cache after each: `torch.cuda.empty_cache()`
- Combine individual masks into composite (bitwise OR)
- Return list of TressMask objects + composite mask

**Remove obsolete functions**:

- Remove `SamAutomaticMaskGenerator` usage
- Remove old full-image downsampling approach
- Keep helper functions for mask utilities

## Phase 4: Analysis Pipeline Integration

### Update `src/analysis.py`

**Modify `analyze_image()` function**:

- After calibration, call `detect_tress_regions()` from tress_detector
- Log: "Detected {len(boxes)} tresses"
- Pass tress_boxes to segmentation module
- Call `segment_all_tresses()` instead of old `segment_tresses()`
- Calculate surface area for each tress using individual masks
- Store per-tress results with tress_id

**Add visualization outputs**:

- Save tress detection image: "tress_detection_{name}.jpg"
- Update segmentation overlay to show tress IDs
- Display crop dimensions in logs

### Update `src/batch_processor.py`

**Modify `process_time_series()`**:

- Remove `max_processing_dim` parameter (no longer relevant)
- Pass tress detection results to analysis
- Ensure per-tress tracking across time points
- Update progress logging to show tress crop sizes

## Phase 5: Testing & Validation

### Test sequence:

1. Test tress detection only - verify all tresses found correctly
2. Test SAM 2 loading and single-crop segmentation
3. Test full pipeline with multiple tresses
4. Verify VRAM usage stays under 4GB
5. Check frizz detail preservation vs old method

### Expected performance:

- Processing time: 15-28 seconds per tress
- Memory: < 4GB VRAM per crop
- Quality: 80-90% frizz detail preservation vs 20% in old method

### To-dos

- [ ] Update requirements.txt and create download_sam2_model.py with auto-download functionality
- [ ] Create src/tress_detector.py with OpenCV-based detection and visualization functions
- [ ] Refactor src/segmentation.py to use SAM 2 predictor with crop-based processing
- [ ] Update src/analysis.py to integrate tress detection and new segmentation workflow
- [ ] Update src/batch_processor.py to support new pipeline and remove obsolete parameters
- [ ] Test the complete pipeline with test images and validate memory usage and quality improvements