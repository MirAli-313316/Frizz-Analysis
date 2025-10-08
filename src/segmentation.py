"""
Segmentation module for detecting hair tresses using SAM 2 with crop-based processing.

This module uses Meta's SAM 2 (Segment Anything Model 2) with point-prompted 
segmentation on individual tress crops at high resolution to preserve fine frizz details.

Hardware Support:
- NVIDIA GeForce RTX 3050 Ti (4GB VRAM) - uses CUDA
- Fallback to CPU if GPU unavailable

Key Optimization:
- Processes each tress crop individually at 1000-2500px resolution
- Avoids aggressive full-image downsampling that loses frizz details
- Uses point prompts for targeted segmentation (more efficient than automatic)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SAM 2 model cache
_sam2_predictor = None
_device = None

# Processing parameters
MAX_CROP_DIMENSION = 2500  # Crops larger than this get resized to 2048px
TARGET_LARGE_CROP_SIZE = 2048  # Target size for large crops
POINTS_PER_SIDE = 48  # Dense grid for point prompts
DARK_PIXEL_THRESHOLD = 150  # Pixels darker than this are likely hair


@dataclass
class TressMask:
    """Container for individual tress segmentation data."""
    
    id: int
    mask: np.ndarray  # Binary mask (height, width) in full image coordinates
    bbox: Tuple[int, int, int, int]  # (x, y, width, height) in full image
    area_pixels: float
    center: Tuple[int, int]  # (x, y)
    confidence: float
    crop_size: Tuple[int, int]  # Original crop dimensions (w, h)
    
    def __repr__(self):
        return (f"TressMask(id={self.id}, area={self.area_pixels:.0f}px, "
                f"bbox={self.bbox}, crop={self.crop_size})")


@dataclass
class SegmentationResult:
    """Container for segmentation results."""
    
    tresses: List[TressMask]
    composite_mask: np.ndarray  # Combined mask of all tresses
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


def load_sam2_model(model_checkpoint: Optional[str] = None):
    """
    Load SAM 2 model with caching.
    
    Args:
        model_checkpoint: Path to model checkpoint file
                         Default: models/sam2_hiera_large.pt
    
    Returns:
        SAM2ImagePredictor instance
    """
    global _sam2_predictor
    
    if _sam2_predictor is not None:
        logger.info("Using cached SAM 2 model")
        return _sam2_predictor
    
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        device = get_device()
        
        # Determine checkpoint path
        if model_checkpoint is None:
            checkpoint_dir = Path("models")
            checkpoint_dir.mkdir(exist_ok=True)
            
            model_checkpoint = checkpoint_dir / "sam2_hiera_large.pt"
            
            if not model_checkpoint.exists():
                logger.error(f"Model checkpoint not found: {model_checkpoint}")
                logger.error("Please download the SAM 2 checkpoint:")
                logger.error(f"  Run: python download_sam2_model.py")
                logger.error(f"  Or manually download from:")
                logger.error(f"  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
                logger.error(f"  Save to: {model_checkpoint}")
                raise FileNotFoundError(f"SAM 2 checkpoint not found: {model_checkpoint}")
        
        logger.info(f"Loading SAM 2 model from {model_checkpoint}")
        
        # Load SAM 2 predictor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        sam2_checkpoint = str(model_checkpoint)
        
        # The config name should include the full path from the sam2 package root
        # Format: "configs/sam2/sam2_hiera_l" (without .yaml extension)
        model_cfg_name = "configs/sam2/sam2_hiera_l"
        
        logger.info(f"Using config: {model_cfg_name}")
        
        # Build the model - hydra will find the config in the sam2 package
        sam2_model = build_sam2(
            model_cfg_name,
            sam2_checkpoint,
            device=device,
            apply_postprocessing=False
        )
        predictor = SAM2ImagePredictor(sam2_model)
        
        _sam2_predictor = predictor
        
        logger.info("SAM 2 model loaded successfully")
        return predictor
    
    except ImportError as e:
        logger.error("sam2 package not installed")
        logger.error("Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        raise e
    except Exception as e:
        logger.error(f"Error loading SAM 2 model: {e}")
        raise e


def generate_point_prompts(
    crop: np.ndarray,
    grid_size: int = POINTS_PER_SIDE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate point prompts for SAM 2 by creating a grid and filtering to dark pixels.
    
    Creates an evenly-spaced grid of points across the crop, then keeps only
    points that fall on dark pixels (likely hair regions).
    
    Args:
        crop: RGB image crop
        grid_size: Number of points per side (creates grid_size x grid_size grid)
    
    Returns:
        Tuple of (input_points, input_labels)
        - input_points: (N, 2) array of [x, y] coordinates
        - input_labels: (N,) array of 1s (all positive prompts)
    """
    height, width = crop.shape[:2]
    
    # Create evenly-spaced grid
    spacing_y = max(1, height // grid_size)
    spacing_x = max(1, width // grid_size)
    
    points = []
    
    # Generate grid points starting from half-spacing to center the grid
    for y in range(spacing_y // 2, height, spacing_y):
        for x in range(spacing_x // 2, width, spacing_x):
            # Check if this pixel is dark (likely hair)
            pixel_value = crop[y, x].mean()
            if pixel_value < DARK_PIXEL_THRESHOLD:
                points.append([x, y])
    
    # Fallback: if too few points found, use center point
    if len(points) < 5:
        logger.warning(f"Only {len(points)} dark points found, using center point as fallback")
        center_x, center_y = width // 2, height // 2
        points = [[center_x, center_y]]
    
    input_points = np.array(points, dtype=np.float32)
    input_labels = np.ones(len(points), dtype=np.int32)  # All positive prompts
    
    logger.debug(f"Generated {len(points)} point prompts from {grid_size}x{grid_size} grid")
    
    return input_points, input_labels


def segment_single_tress(
    image: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    tress_id: int,
    predictor
) -> TressMask:
    """
    Segment a single tress from its cropped region using SAM 2 point prompts.
    
    This function:
    1. Extracts the crop from the full image
    2. Intelligently resizes if needed (>2500px -> 2048px)
    3. Generates point prompts on dark pixels
    4. Runs SAM 2 prediction
    5. Scales mask back if resized
    6. Places mask in full image coordinates
    
    Args:
        image: Full RGB image
        crop_box: (x, y, w, h) bounding box for this tress
        tress_id: Identifier for this tress
        predictor: SAM2ImagePredictor instance
    
    Returns:
        TressMask with segmentation results
    """
    x, y, w, h = crop_box
    
    # Extract crop
    crop = image[y:y+h, x:x+w].copy()
    crop_height, crop_width = crop.shape[:2]
    
    logger.info(f"Processing tress {tress_id}: crop size {crop_width}x{crop_height} px")
    
    # Smart resizing: only resize if crop is very large
    scale_factor = 1.0
    processing_crop = crop
    
    max_dim = max(crop_height, crop_width)
    if max_dim > MAX_CROP_DIMENSION:
        # Resize to target size
        scale_factor = TARGET_LARGE_CROP_SIZE / max_dim
        new_width = int(crop_width * scale_factor)
        new_height = int(crop_height * scale_factor)
        if max_dim > MAX_CROP_DIMENSION:
            # Resize to target size
            processing_crop = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"  Resized crop to {new_width}x{new_height} for processing (scale: {scale_factor:.3f})")
    else:
        logger.info(f"  Processing at native resolution (no resize needed)")

        # Generate point prompts on the processing crop
        input_points, input_labels = generate_point_prompts(processing_crop)
    
    # Run SAM 2 prediction
    try:
        with torch.inference_mode():
            # Set the image
            predictor.set_image(processing_crop)
            
            # Predict mask
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False  # Single best mask
            )
            
            # Extract the mask (first and only mask)
            crop_mask = masks[0]  # Shape: (h, w)
            confidence = float(scores[0])
    
    except Exception as e:
        logger.error(f"Error during SAM 2 prediction for tress {tress_id}: {e}")
        # Return empty mask as fallback
        crop_mask = np.zeros((crop_height, crop_width), dtype=bool)
        confidence = 0.0
    
    # Scale mask back to original crop size if we resized
    if scale_factor < 1.0:
        crop_mask = cv2.resize(
            crop_mask.astype(np.uint8),
            (crop_width, crop_height),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    
    # Create full-image mask and place crop mask at correct position
    full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    full_mask[y:y+h, x:x+w] = crop_mask
    
    # Calculate metrics
    area_pixels = float(np.sum(crop_mask))
    
    # Calculate center (in full image coordinates)
    if area_pixels > 0:
        coords = np.argwhere(crop_mask)
        center_rel = coords.mean(axis=0)
        center = (int(x + center_rel[1]), int(y + center_rel[0]))  # (x, y)
    else:
        center = (x + w // 2, y + h // 2)
    
    logger.info(f"  Segmented {area_pixels:.0f} pixels, confidence: {confidence:.3f}")
    
    return TressMask(
        id=tress_id,
        mask=full_mask,
        bbox=crop_box,
        area_pixels=area_pixels,
        center=center,
        confidence=confidence,
        crop_size=(crop_width, crop_height)
    )


def segment_all_tresses(
    image: np.ndarray,
    tress_boxes: List[Tuple[int, int, int, int]],
    predictor = None
) -> SegmentationResult:
    """
    Segment all detected tresses using crop-based SAM 2 processing.
    
    Processes each tress crop individually at high resolution, then combines
    the results into a composite mask.
    
    Args:
        image: Full RGB image
        tress_boxes: List of (x, y, w, h) bounding boxes
        predictor: SAM2ImagePredictor (will load if None)
    
    Returns:
        SegmentationResult with individual tress masks and composite
    """
    start_time = time.time()
    
    # Load predictor if not provided
    if predictor is None:
        predictor = load_sam2_model()
    
    device = get_device()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SEGMENTING {len(tress_boxes)} TRESSES WITH SAM 2")
    logger.info(f"{'='*70}")
    
    # Process each tress
    tress_masks = []
    
    for i, crop_box in enumerate(tress_boxes, 1):
        logger.info(f"\nTress {i}/{len(tress_boxes)}:")
        
        tress_mask = segment_single_tress(image, crop_box, i, predictor)
        tress_masks.append(tress_mask)
        
        # Clear GPU cache after each tress
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Create composite mask (combine all individual masks)
    composite_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    for tress_mask in tress_masks:
        composite_mask = composite_mask | tress_mask.mask
    
    processing_time = time.time() - start_time
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SEGMENTATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Tresses segmented: {len(tress_masks)}")
    logger.info(f"Total processing time: {processing_time:.2f}s")
    logger.info(f"Average per tress: {processing_time/len(tress_masks):.2f}s")
    logger.info(f"{'='*70}\n")
    
    return SegmentationResult(
        tresses=tress_masks,
        composite_mask=composite_mask,
        image_shape=image.shape,
        device_used=device,
        processing_time=processing_time
    )
    

def visualize_segmentation(
    image: np.ndarray,
    segmentation_result: SegmentationResult,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create visualization of segmentation results with color-coded tresses.
    
    Args:
        image: Original RGB image
        segmentation_result: SegmentationResult from segment_all_tresses
        alpha: Transparency of overlay (0=transparent, 1=opaque)
    
    Returns:
        Annotated image with colored masks and labels
    """
    viz_image = image.copy()
    
    # Create color map for different tresses
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 255, 128),  # Spring green
        (255, 0, 128),  # Rose
    ]
    
    # Draw each tress with different color
    overlay = image.copy()
    
    for tress in segmentation_result.tresses:
        color = colors[(tress.id - 1) % len(colors)]
        
        # Apply color to mask region
        overlay[tress.mask] = overlay[tress.mask] * (1 - alpha) + np.array(color) * alpha
        
        # Draw bounding box
        x, y, w, h = tress.bbox
        cv2.rectangle(viz_image, (x, y), (x + w, y + h), color, 2)
        
        # Add label
        label = f"T{tress.id}"
        cv2.putText(viz_image, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Blend overlay
    viz_image = cv2.addWeighted(viz_image, 1 - alpha, overlay, alpha, 0)
    
    return viz_image