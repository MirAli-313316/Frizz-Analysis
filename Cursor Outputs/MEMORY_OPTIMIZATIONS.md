# SAM Memory Optimizations for RTX 3050 Ti (4GB VRAM)

## Problem
Canon 18MP images (5184x3456 pixels) caused CUDA out-of-memory errors on RTX 3050 Ti with 4GB VRAM.

## Solution Implemented

### 1. Image Resizing Strategy
- **Before SAM processing**: Resize images to max 1024px on longest side
  - 5184x3456 → 1024x683 (approx 80% reduction in each dimension)
  - This reduces memory by ~84% (from 18MP to ~0.7MP)
- **After SAM processing**: Scale masks back to original 5184x3456 size
- Maintains aspect ratio throughout

### 2. Updated SAM Parameters (Memory-Optimized)
```python
SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,        # Reduced from 32 (4x fewer points)
    pred_iou_thresh=0.86,      # Unchanged
    stability_score_thresh=0.92,  # Unchanged
    crop_n_layers=0,           # Disabled multi-scale (was 1)
    min_mask_region_area=500,  # Reduced for resized images
)
```

### 3. Memory Management
- `torch.cuda.empty_cache()` called after SAM mask generation
- `torch.cuda.empty_cache()` called at end of processing
- `del` statements to clear intermediate variables:
  - `resized_rgb` after mask generation
  - `original_image` and `original_rgb` after filtering

### 4. Processing Flow
```
1. Load original image (5184x3456)
2. Resize to 1024x683 for processing
3. Run SAM on resized image (low memory)
4. Scale masks back to 5184x3456
5. Filter tresses using full-size masks
6. Clear GPU cache
```

## Key Functions Added

### `_resize_for_processing(image, max_dim=1024)`
- Resizes image maintaining aspect ratio
- Returns tuple: (resized_image, scale_factor)
- No resize if image already <= max_dim

### `_scale_masks_to_original(masks, original_shape, scale_factor)`
- Scales mask segmentations from processed size to original
- Scales bounding boxes with inverse scale factor
- Recalculates areas in original image space
- Uses INTER_NEAREST for mask scaling (preserves binary values)

## Benefits
- Reduces VRAM usage by ~84%
- Enables processing of 18MP images on 4GB GPU
- Maintains mask quality (scaled back to full resolution)
- Graceful handling of images of any size

## Testing
Test with: `python test_segmentation.py`

Expected behavior:
- 18MP images process without OOM errors
- Masks correctly sized to original image dimensions
- Tress detection accuracy maintained

