# Known Issues & Workarounds

## GPU Memory Limitations
**Issue**: Direct processing of 18MP images exceeds 4GB VRAM
**Status**: ✅ RESOLVED via image resizing pipeline
**Solution**: Process at 1024px, scale masks back to original
**Implementation**: See `_resize_for_processing()` and `_scale_masks_to_original()` in segmentation.py

## Quarter Detection on Translucent Plastic
**Issue**: Edge detection affected by translucent mounting
**Status**: ✅ RESOLVED via multi-strategy detection
**Solution**: Three-tier detection with progressive parameter relaxation
**Implementation**: See `_detect_circles_multi_strategy()` in calibration.py

## Frame Detection in Segmentation
**Issue**: Rectangular frames/boxes detected as tresses
**Status**: ✅ RESOLVED via shape-based filtering
**Solution**: 
- Rectangularity threshold (> 0.85) filters rectangular frames
- Narrow strip detection (aspect ratio > 6.5) catches frame edges
- Comprehensive diagnostic logging for debugging
**Implementation**: See `_is_frame_or_background()` in segmentation.py
**Date Resolved**: 2025-10-02

## Merged Tress Detection
**Issue**: SAM detects multiple separate tresses as single merged object
**Status**: ✅ RESOLVED via automatic splitting
**Solution**:
- Morphological operations to separate connected regions
- Connected components analysis to identify individual tresses
- Automatic splitting for masks > 850k pixels (~50 cm²)
**Implementation**: See `_split_merged_tresses()` in segmentation.py
**Date Resolved**: 2025-10-02

## Cursor Chat Memory Errors
**Issue**: Cursor chat freezes when processing large models
**Workaround**: Restart chat, run tests directly in terminal
**Note**: Does not affect code execution, only chat interface