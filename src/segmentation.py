"""
Segmentation module for detecting hair tresses using SAM (Segment Anything Model).

This module uses Meta's Segment Anything Model to automatically detect and segment
hair tresses in images. The tresses are dark vertical objects against a white background,
typically arranged in rows.

Hardware Support:
- NVIDIA GeForce RTX 3050 Ti (4GB VRAM) - uses CUDA
- Fallback to CPU if GPU unavailable
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SAM model cache
_sam_model = None
_sam_predictor = None
_device = None


@dataclass
class TressMask:
    """Container for individual tress segmentation data."""
    
    id: int
    mask: np.ndarray  # Binary mask (height, width)
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    area_pixels: float
    center: Tuple[int, int]  # (x, y)
    confidence: float
    aspect_ratio: float  # height/width
    
    def __repr__(self):
        return (f"TressMask(id={self.id}, area={self.area_pixels:.0f}px, "
                f"bbox={self.bbox}, confidence={self.confidence:.2f})")


@dataclass
class SegmentationResult:
    """Container for segmentation results."""
    
    tresses: List[TressMask]
    image_shape: Tuple[int, int, int]
    device_used: str
    processing_time: float
    
    def __repr__(self):
        return (f"SegmentationResult(tresses={len(self.tresses)}, "
                f"device={self.device_used}, time={self.processing_time:.2f}s)")


def get_device() -> str:
    """
    Determine the best available device (CUDA GPU or CPU).
    
    Returns:
        Device string: 'cuda' or 'cpu'
    """
    global _device
    
    if _device is not None:
        return _device
    
    if torch.cuda.is_available():
        _device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info("Using CUDA acceleration")
    else:
        _device = 'cpu'
        logger.warning("CUDA not available. Using CPU (this will be slower)")
        logger.warning("To use GPU: Ensure PyTorch with CUDA support is installed")
    
    return _device


def load_sam_model(model_type: str = 'vit_b', model_checkpoint: Optional[str] = None) -> Tuple:
    """
    Load SAM model with caching.
    
    Args:
        model_type: Model variant ('vit_h', 'vit_l', 'vit_b')
                   vit_b is recommended for 4GB VRAM
        model_checkpoint: Path to model checkpoint file
    
    Returns:
        Tuple of (sam_model, mask_generator)
    """
    global _sam_model, _sam_predictor
    
    if _sam_model is not None:
        logger.info("Using cached SAM model")
        return _sam_model, _sam_predictor
    
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        
        device = get_device()
        
        # Determine checkpoint path
        if model_checkpoint is None:
            checkpoint_dir = Path("models")
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_filename = f"sam_{model_type}.pth"
            model_checkpoint = checkpoint_dir / checkpoint_filename
            
            if not model_checkpoint.exists():
                logger.error(f"Model checkpoint not found: {model_checkpoint}")
                logger.error("Please download the SAM checkpoint:")
                logger.error(f"  URL: https://dl.fbaipublicfiles.com/segment_anything/sam_{model_type}.pth")
                logger.error(f"  Save to: {model_checkpoint}")
                raise FileNotFoundError(f"SAM checkpoint not found: {model_checkpoint}")
        
        logger.info(f"Loading SAM model: {model_type} from {model_checkpoint}")
        sam = sam_model_registry[model_type](checkpoint=str(model_checkpoint))
        sam.to(device=device)
        
        # Create automatic mask generator with memory-optimized parameters
        # Tuned for RTX 3050 Ti (4GB VRAM) with 18MP images
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,  # Reduced from 32 to save memory
            pred_iou_thresh=0.86,  # Quality threshold
            stability_score_thresh=0.92,  # Stability threshold
            crop_n_layers=0,  # Disabled multi-scale cropping to save memory
            crop_n_points_downscale_factor=2,
            min_mask_region_area=500,  # Lower threshold for resized images
        )
        
        _sam_model = sam
        _sam_predictor = mask_generator
        
        logger.info("SAM model loaded successfully")
        return sam, mask_generator
    
    except ImportError as e:
        logger.error("segment-anything package not installed")
        logger.error("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
        raise e
    except Exception as e:
        logger.error(f"Error loading SAM model: {e}")
        raise e


def _resize_for_processing(
    image: np.ndarray,
    max_dim: int = 1024
) -> Tuple[np.ndarray, float]:
    """
    Resize image for SAM processing to reduce memory usage.
    
    Maintains aspect ratio while ensuring longest dimension <= max_dim.
    
    Args:
        image: RGB image array
        max_dim: Maximum dimension (width or height)
    
    Returns:
        Tuple of (resized_image, scale_factor)
        scale_factor is the ratio: resized_dim / original_dim
    """
    height, width = image.shape[:2]
    longest_dim = max(height, width)
    
    if longest_dim <= max_dim:
        # No resizing needed
        return image, 1.0
    
    # Calculate scale factor
    scale_factor = max_dim / longest_dim
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized, scale_factor


def _scale_masks_to_original(
    masks: List[Dict],
    original_shape: Tuple[int, int],
    scale_factor: float
) -> List[Dict]:
    """
    Scale masks and bounding boxes from processed image back to original size.
    
    Args:
        masks: List of mask dictionaries from SAM
        original_shape: (height, width) of original image
        scale_factor: Scale factor used for resizing (processed / original)
    
    Returns:
        List of mask dictionaries with scaled masks and bboxes
    """
    orig_height, orig_width = original_shape
    inverse_scale = 1.0 / scale_factor
    
    scaled_masks = []
    
    for mask_data in masks:
        # Get original mask (in processed image size)
        small_mask = mask_data['segmentation']
        
        # Scale mask back to original size
        full_mask = cv2.resize(
            small_mask.astype(np.uint8),
            (orig_width, orig_height),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        # Scale bounding box
        x, y, w, h = mask_data['bbox']
        scaled_bbox = (
            int(x * inverse_scale),
            int(y * inverse_scale),
            int(w * inverse_scale),
            int(h * inverse_scale)
        )
        
        # Calculate area in original image
        scaled_area = int(np.sum(full_mask))
        
        # Create scaled mask data
        scaled_data = {
            'segmentation': full_mask,
            'area': scaled_area,
            'bbox': scaled_bbox,
            'predicted_iou': mask_data.get('predicted_iou', 0.0),
            'stability_score': mask_data.get('stability_score', 0.0),
        }
        
        scaled_masks.append(scaled_data)
    
    return scaled_masks


def _calculate_rectangularity(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """
    Calculate how rectangular a mask is (0 = not rectangular, 1 = perfect rectangle).
    
    Rectangularity = mask_area / bbox_area
    A perfect rectangle fills its bounding box completely (ratio = 1.0)
    
    Args:
        mask: Binary mask
        bbox: Bounding box (x, y, width, height)
    
    Returns:
        Rectangularity score (0.0 to 1.0)
    """
    x, y, w, h = bbox
    bbox_area = w * h
    mask_area = np.sum(mask)
    
    if bbox_area == 0:
        return 0.0
    
    rectangularity = mask_area / bbox_area
    return rectangularity


def _calculate_solidity(mask: np.ndarray) -> float:
    """
    Calculate solidity: ratio of mask area to convex hull area.
    
    Low solidity (< 0.2) indicates hollow objects like frames.
    High solidity (> 0.8) indicates solid objects like tresses.
    
    Args:
        mask: Binary mask
    
    Returns:
        Solidity score (0.0 to 1.0)
    """
    # Find contours
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 1.0
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate areas
    mask_area = np.sum(mask)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return 1.0
    
    solidity = mask_area / hull_area
    return solidity


def _touches_image_edges(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    margin: int = 10
) -> int:
    """
    Count how many image edges the bounding box touches.
    
    Args:
        bbox: Bounding box (x, y, width, height)
        image_shape: (height, width) of image
        margin: Pixel margin for edge detection
    
    Returns:
        Number of edges touched (0-4)
    """
    x, y, w, h = bbox
    img_height, img_width = image_shape
    
    edges_touched = 0
    
    # Check each edge
    if x <= margin:  # Left edge
        edges_touched += 1
    if y <= margin:  # Top edge
        edges_touched += 1
    if x + w >= img_width - margin:  # Right edge
        edges_touched += 1
    if y + h >= img_height - margin:  # Bottom edge
        edges_touched += 1
    
    return edges_touched


def _analyze_mask_color(image: np.ndarray, mask: np.ndarray) -> str:
    """
    Analyze the dominant color of a mask to help identify frames.
    
    Cyan/blue frames are common in hair tress setups.
    
    Args:
        image: BGR image
        mask: Binary mask
    
    Returns:
        Color description string
    """
    # Get masked pixels
    masked_pixels = image[mask]
    
    if len(masked_pixels) == 0:
        return "empty"
    
    # Calculate mean color in BGR
    mean_bgr = np.mean(masked_pixels, axis=0)
    b, g, r = mean_bgr
    
    # Convert to HSV for better color detection
    bgr_sample = np.uint8([[mean_bgr]])
    hsv_sample = cv2.cvtColor(bgr_sample, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv_sample
    
    # Classify color
    if s < 30:  # Low saturation = grayscale
        if v > 180:
            return "white/bright"
        elif v < 50:
            return "black/dark"
        else:
            return "gray"
    
    # Check for cyan/blue (common frame color)
    # Cyan: hue 85-100, Blue: hue 100-130
    if 85 <= h <= 130 and s > 30:
        return f"cyan/blue (H={h}, S={s})"
    
    # Other colors
    if h < 15 or h > 165:
        return f"red (H={h})"
    elif 15 <= h < 35:
        return f"yellow (H={h})"
    elif 35 <= h < 85:
        return f"green (H={h})"
    elif 130 <= h < 165:
        return f"purple (H={h})"
    
    return f"color(H={h}, S={s}, V={v})"


def _split_merged_tresses(
    mask: np.ndarray,
    min_component_area: int = 50000
) -> List[np.ndarray]:
    """
    Split a large mask that may contain multiple merged tresses.
    
    Uses morphological operations and connected components analysis to separate
    tresses that SAM detected as a single object but are actually multiple tresses
    separated by gaps.
    
    Args:
        mask: Binary mask that may contain multiple tresses
        min_component_area: Minimum area for a valid component (default 50k pixels)
    
    Returns:
        List of individual masks (may be just [original_mask] if no split found)
    """
    # Convert to uint8 for OpenCV
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Apply morphological opening to break weak connections between tresses
    # Kernel size chosen to break small bridges but preserve tress structure
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # Additional erosion to further separate touching tresses
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_mask = cv2.erode(opened_mask, erode_kernel, iterations=2)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        eroded_mask, connectivity=8
    )
    
    # Extract individual components (skip background label 0)
    individual_masks = []
    
    for label in range(1, num_labels):
        component_mask = (labels == label)
        component_area = stats[label, cv2.CC_STAT_AREA]
        
        # Filter out small noise components
        if component_area < min_component_area:
            continue
        
        # Dilate back to recover original tress boundaries
        component_uint8 = (component_mask * 255).astype(np.uint8)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        recovered_mask = cv2.dilate(component_uint8, dilate_kernel, iterations=2)
        
        # Intersect with original mask to avoid growing beyond original boundaries
        final_mask = np.logical_and(recovered_mask > 0, mask)
        
        individual_masks.append(final_mask)
    
    # Return original mask if no valid split found
    if len(individual_masks) == 0:
        return [mask]
    
    # Only return split if we found multiple components
    if len(individual_masks) == 1:
        # Check if the single component is significantly smaller than original
        # If so, we may have over-eroded, use original
        single_area = np.sum(individual_masks[0])
        original_area = np.sum(mask)
        if single_area < original_area * 0.7:
            return [mask]
    
    return individual_masks


def _is_frame_or_background(
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    area: float,
    image_shape: Tuple[int, int],
    rectangularity_threshold: float = 0.85,
    solidity_threshold: float = 0.15,
    edge_size_threshold: float = 0.5
) -> Tuple[bool, str]:
    """
    Detect if a mask is likely a frame, box, or background rather than a tress.
    
    Detection criteria:
    1. High rectangularity (> 0.85) = likely a rectangular frame
    2. Low solidity (< 0.15) = hollow rectangle/frame
    3. Touches 3+ edges AND large size = likely frame or background
    4. Narrow strips (width < 15% height or height < 15% width) with high rectangularity
    
    Args:
        mask: Binary mask
        bbox: Bounding box (x, y, width, height)
        area: Mask area in pixels
        image_shape: (height, width) of image
        rectangularity_threshold: Threshold for rectangular detection (default 0.85)
        solidity_threshold: Threshold for hollow detection (default 0.15)
        edge_size_threshold: Size threshold for edge-touching objects (default 0.5)
    
    Returns:
        Tuple of (is_frame, reason)
    """
    x, y, w, h = bbox
    img_height, img_width = image_shape
    
    # Calculate metrics
    rectangularity = _calculate_rectangularity(mask, bbox)
    solidity = _calculate_solidity(mask)
    edges_touched = _touches_image_edges(bbox, image_shape)
    
    # Check if dimensions are very large (>50% of image)
    width_ratio = w / img_width
    height_ratio = h / img_height
    is_large = (width_ratio > edge_size_threshold and height_ratio > edge_size_threshold)
    
    # Check for narrow strips (frame edges)
    # If width is < 15% of height (tall narrow strip) OR height < 15% of width (wide flat strip)
    aspect_ratio = h / w if w > 0 else 0
    is_narrow_strip = aspect_ratio > 6.5 or (1 / aspect_ratio > 6.5 if aspect_ratio > 0 else False)
    
    # Detection logic
    if rectangularity > rectangularity_threshold:
        return True, f"rectangular frame (rectangularity={rectangularity:.2f})"
    
    # Lower threshold for narrow strips (frame edges are typically narrow and rectangular)
    if is_narrow_strip and rectangularity > 0.75:
        return True, f"narrow frame strip (rectangularity={rectangularity:.2f}, aspect={aspect_ratio:.1f})"
    
    if solidity < solidity_threshold:
        return True, f"hollow frame (solidity={solidity:.2f})"
    
    if edges_touched >= 3 and is_large:
        return True, f"edge-touching large object (edges={edges_touched}, size={width_ratio:.1%}x{height_ratio:.1%})"
    
    return False, ""


def segment_tresses(
    image_path: str,
    exclude_region: Optional[Tuple[int, int, int, int]] = None,
    min_tress_area: int = 200000,
    max_tress_area: int = 1800000,
    brightness_threshold: int = 150,
    max_processing_dim: int = 1024
) -> SegmentationResult:
    """
    Segment hair tresses from image using SAM.
    
    Memory-optimized for RTX 3050 Ti (4GB VRAM):
    - Resizes large images before SAM processing
    - Scales masks back to original dimensions
    - Clears GPU cache after processing
    
    Frame detection:
    - Filters out rectangular frames/boxes (high rectangularity)
    - Removes hollow rectangles (low solidity)
    - Excludes edge-touching large objects
    
    Args:
        image_path: Path to image file
        exclude_region: Region to exclude (x, y, width, height) - typically quarter area
        min_tress_area: Minimum area in pixels for valid tress (default ~200k = 10 cm²)
        max_tress_area: Maximum area in pixels for valid tress (default ~1.8M = 90 cm²)
        brightness_threshold: Maximum average brightness for tress (darker than background)
        max_processing_dim: Maximum dimension for SAM processing (default 1024 for 4GB VRAM)
    
    Returns:
        SegmentationResult containing detected tresses
    """
    import time
    start_time = time.time()
    
    logger.info(f"Loading image: {image_path}")
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = original_image.shape[:2]
    logger.info(f"Original image dimensions: {orig_width}x{orig_height} pixels")
    
    # Resize image for SAM processing to save memory
    resized_rgb, scale_factor = _resize_for_processing(original_rgb, max_processing_dim)
    proc_height, proc_width = resized_rgb.shape[:2]
    
    if scale_factor < 1.0:
        logger.info(f"Resized for processing: {proc_width}x{proc_height} pixels (scale: {scale_factor:.3f})")
    else:
        logger.info(f"Image size OK for processing: {proc_width}x{proc_height} pixels")
    
    # Load SAM model
    sam, mask_generator = load_sam_model()
    device = get_device()
    
    # Generate masks on resized image
    logger.info("Generating masks with SAM (this may take a moment)...")
    try:
        masks = mask_generator.generate(resized_rgb)
        logger.info(f"Generated {len(masks)} candidate masks")
    finally:
        # Clear GPU cache immediately after SAM processing
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Scale masks back to original image size
    if scale_factor < 1.0:
        logger.info("Scaling masks to original image size...")
        masks = _scale_masks_to_original(masks, (orig_height, orig_width), scale_factor)
    
    # Clear resized image from memory
    del resized_rgb
    
    # Filter masks to find tresses (using original image for brightness check)
    tresses = []
    filtered_count = {'quarter': 0, 'too_small': 0, 'too_large': 0, 'frame': 0, 'too_bright': 0, 'split': 0}
    
    logger.info(f"\n{'='*70}")
    logger.info(f"DETAILED MASK ANALYSIS - Filtering {len(masks)} candidate masks")
    logger.info(f"{'='*70}")
    
    # Threshold for splitting (50 cm² ≈ 850,000 pixels with typical calibration)
    split_threshold = 850000
    
    for idx, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        area = mask_data['area']
        bbox = mask_data['bbox']  # (x, y, w, h)
        x, y, w, h = bbox
        
        # Calculate all metrics for diagnostic logging
        rectangularity = _calculate_rectangularity(mask, bbox)
        solidity = _calculate_solidity(mask)
        avg_brightness = _calculate_average_brightness(original_image, mask)
        edges_touched = _touches_image_edges(bbox, (orig_height, orig_width))
        
        # Analyze color (check for cyan/blue frame)
        color_hint = _analyze_mask_color(original_image, mask)
        
        # Log complete diagnostic info for ALL masks
        logger.info(f"\nMask {idx+1}:")
        logger.info(f"  Area: {area:,} pixels")
        logger.info(f"  BBox: {w}x{h} at ({x}, {y})")
        logger.info(f"  Rectangularity: {rectangularity:.3f}")
        logger.info(f"  Solidity: {solidity:.3f}")
        logger.info(f"  Brightness: {avg_brightness:.1f}")
        logger.info(f"  Edges touched: {edges_touched}/4")
        logger.info(f"  Color hint: {color_hint}")
        
        # Skip if mask overlaps with exclude region (quarter)
        if exclude_region is not None:
            if _overlaps_with_region(mask, exclude_region):
                filtered_count['quarter'] += 1
                logger.info(f"  ❌ FILTERED: Overlaps quarter region")
                continue
        
        # Filter by area range (200k - 1.8M pixels ≈ 10-90 cm²)
        if area < min_tress_area:
            filtered_count['too_small'] += 1
            logger.info(f"  ❌ FILTERED: Too small ({area:,} < {min_tress_area:,})")
            continue
        
        if area > max_tress_area:
            filtered_count['too_large'] += 1
            logger.info(f"  ❌ FILTERED: Too large ({area:,} > {max_tress_area:,})")
            continue
        
        # Check if this is a frame or background
        is_frame, frame_reason = _is_frame_or_background(
            mask, bbox, area, (orig_height, orig_width)
        )
        
        if is_frame:
            filtered_count['frame'] += 1
            logger.info(f"  ❌ FILTERED: {frame_reason}")
            continue
        
        # Filter by brightness (tresses are dark) - use original image
        if avg_brightness > brightness_threshold:
            filtered_count['too_bright'] += 1
            logger.info(f"  ❌ FILTERED: Too bright ({avg_brightness:.1f} > {brightness_threshold})")
            continue
        
        # Check if this mask might contain multiple merged tresses
        # If area > 850k pixels (~50 cm²), attempt to split
        masks_to_process = [mask]
        
        if area > split_threshold:
            logger.info(f"  ⚠️  Large mask detected ({area:,} pixels > {split_threshold:,})")
            logger.info(f"  → Attempting to split merged tresses...")
            
            split_masks = _split_merged_tresses(mask, min_component_area=200000)
            
            if len(split_masks) > 1:
                filtered_count['split'] += 1
                logger.info(f"  ✓ Successfully split into {len(split_masks)} separate tresses!")
                masks_to_process = split_masks
            else:
                logger.info(f"  → No split found, keeping as single tress")
        
        # Process each mask (either original or split components)
        for component_idx, component_mask in enumerate(masks_to_process):
            # Recalculate properties for the component
            component_area = int(np.sum(component_mask))
            
            # Find bounding box for component
            ys, xs = np.where(component_mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            
            comp_x = int(np.min(xs))
            comp_y = int(np.min(ys))
            comp_w = int(np.max(xs) - comp_x + 1)
            comp_h = int(np.max(ys) - comp_y + 1)
            comp_bbox = (comp_x, comp_y, comp_w, comp_h)
            
            # Calculate aspect ratio and center
            if comp_w == 0:
                continue
            comp_aspect_ratio = comp_h / comp_w
            comp_center_x = comp_x + comp_w // 2
            comp_center_y = comp_y + comp_h // 2
            
            # Log acceptance
            if len(masks_to_process) > 1:
                logger.info(f"  ✓ ACCEPTED as tress #{len(tresses)+1} (split component {component_idx+1}/{len(masks_to_process)})")
            else:
                logger.info(f"  ✓ ACCEPTED as tress #{len(tresses)+1}")
            
            # Create TressMask object
            tress = TressMask(
                id=len(tresses) + 1,
                mask=component_mask,
                bbox=comp_bbox,
                area_pixels=float(component_area),
                center=(comp_center_x, comp_center_y),
                confidence=float(mask_data.get('predicted_iou', 0.0)),
                aspect_ratio=comp_aspect_ratio
            )
            tresses.append(tress)
    
    # Log filtering summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Filtering summary:")
    logger.info(f"  Quarter overlaps: {filtered_count['quarter']}")
    logger.info(f"  Too small: {filtered_count['too_small']}")
    logger.info(f"  Too large: {filtered_count['too_large']}")
    logger.info(f"  Frames/boxes: {filtered_count['frame']}")
    logger.info(f"  Too bright: {filtered_count['too_bright']}")
    logger.info(f"  Large masks split: {filtered_count['split']}")
    logger.info(f"  Accepted tresses: {len(tresses)}")
    logger.info(f"{'='*70}")
    
    # Clear original images from memory
    del original_image
    del original_rgb
    
    # Sort tresses by position (top to bottom, left to right)
    tresses = _sort_tresses_by_position(tresses)
    
    # Reassign IDs after sorting
    for idx, tress in enumerate(tresses):
        tress.id = idx + 1
    
    processing_time = time.time() - start_time
    logger.info(f"Detected {len(tresses)} tresses in {processing_time:.2f}s")
    
    for tress in tresses:
        logger.info(f"  Tress {tress.id}: {tress.area_pixels:.0f} pixels, "
                   f"aspect ratio {tress.aspect_ratio:.2f}, confidence {tress.confidence:.2f}")
    
    # Final GPU cache clear
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    result = SegmentationResult(
        tresses=tresses,
        image_shape=(orig_height, orig_width, 3),
        device_used=device,
        processing_time=processing_time
    )
    
    return result


def _overlaps_with_region(mask: np.ndarray, region: Tuple[int, int, int, int]) -> bool:
    """
    Check if mask overlaps with specified region.
    
    Args:
        mask: Binary mask array
        region: (x, y, width, height)
    
    Returns:
        True if overlap exists
    """
    x, y, w, h = region
    region_mask = mask[y:y+h, x:x+w]
    return np.any(region_mask)


def _calculate_average_brightness(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate average brightness of masked region.
    
    Args:
        image: BGR image
        mask: Binary mask
    
    Returns:
        Average brightness (0-255)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_pixels = gray[mask]
    if len(masked_pixels) == 0:
        return 255.0
    return float(np.mean(masked_pixels))


def _sort_tresses_by_position(tresses: List[TressMask]) -> List[TressMask]:
    """
    Sort tresses by position: top to bottom, left to right.
    
    This handles the typical 2-row arrangement but works for any layout.
    
    Args:
        tresses: List of TressMask objects
    
    Returns:
        Sorted list of tresses
    """
    if not tresses:
        return tresses
    
    # Sort by y-coordinate (top to bottom), then by x-coordinate (left to right)
    # Use center positions for consistent ordering
    sorted_tresses = sorted(tresses, key=lambda t: (t.center[1], t.center[0]))
    
    return sorted_tresses


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Clean up mask by removing noise and filling holes.
    
    Args:
        mask: Binary mask
        kernel_size: Size of morphological kernel
    
    Returns:
        Cleaned binary mask
    """
    # Convert to uint8 if needed
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to boolean
    return mask > 0


def create_visualization(
    image_path: str,
    segmentation_result: SegmentationResult,
    show_labels: bool = True,
    show_bboxes: bool = True,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create visualization of segmentation results.
    
    Args:
        image_path: Path to original image
        segmentation_result: Segmentation results
        show_labels: Whether to show tress labels
        show_bboxes: Whether to show bounding boxes
        alpha: Transparency of mask overlay (0-1)
    
    Returns:
        Visualization image
    """
    # Load original image
    image = cv2.imread(image_path)
    vis_image = image.copy()
    
    # Generate distinct colors for each tress
    colors = _generate_distinct_colors(len(segmentation_result.tresses))
    
    # Create overlay
    overlay = image.copy()
    
    for tress, color in zip(segmentation_result.tresses, colors):
        # Apply colored mask
        overlay[tress.mask] = color
        
        # Draw bounding box
        if show_bboxes:
            x, y, w, h = tress.bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 3)
        
        # Add label
        if show_labels:
            x, y, w, h = tress.bbox
            label = f"Tress {tress.id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle
            label_x = x + 5
            label_y = y - 10
            cv2.rectangle(
                vis_image,
                (label_x - 5, label_y - text_height - 5),
                (label_x + text_width + 5, label_y + baseline + 5),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                vis_image,
                label,
                (label_x, label_y),
                font,
                font_scale,
                color,
                thickness
            )
    
    # Blend overlay with original image
    vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)
    
    # Add summary text
    summary_text = [
        f"Tresses Detected: {len(segmentation_result.tresses)}",
        f"Device: {segmentation_result.device_used.upper()}",
        f"Processing Time: {segmentation_result.processing_time:.2f}s"
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    y_offset = 50
    
    for i, text in enumerate(summary_text):
        y_pos = y_offset + i * 40
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw background
        cv2.rectangle(
            vis_image,
            (10, y_pos - text_height - 5),
            (20 + text_width, y_pos + baseline + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            vis_image,
            text,
            (15, y_pos),
            font,
            font_scale,
            (0, 255, 0),
            thickness
        )
    
    return vis_image


def _generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate n visually distinct colors.
    
    Args:
        n: Number of colors to generate
    
    Returns:
        List of BGR color tuples
    """
    colors = []
    for i in range(n):
        hue = int(180 * i / n)  # Spread across hue spectrum
        # Create in HSV, convert to BGR
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr)))
    return colors


def create_exclude_region_from_quarter(
    quarter_center: Tuple[int, int],
    quarter_radius: int,
    margin: int = 50
) -> Tuple[int, int, int, int]:
    """
    Create exclude region around detected quarter.
    
    Args:
        quarter_center: (x, y) center of quarter
        quarter_radius: Radius of quarter in pixels
        margin: Additional margin around quarter
    
    Returns:
        Exclude region as (x, y, width, height)
    """
    x, y = quarter_center
    size = (quarter_radius + margin) * 2
    
    exclude_x = max(0, x - quarter_radius - margin)
    exclude_y = max(0, y - quarter_radius - margin)
    
    return (exclude_x, exclude_y, size, size)

