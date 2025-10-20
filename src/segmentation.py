"""
Segmentation module for detecting hair tresses using BiRefNet with preprocessing.

This module uses BiRefNet (salient object detection) for hair tress segmentation
with preprocessing optimization for better frizz detail capture.

Hardware Support:
- NVIDIA GeForce RTX 3050 Ti (4GB VRAM) - uses CUDA
- Fallback to CPU if GPU unavailable

Key Features:
- BiRefNet_lite model for lightweight salient object detection
- CLAHE preprocessing for enhanced contrast and edge detection
- Lower segmentation threshold (0.35) to capture frizzy edges
- Single-pass processing (no need for point prompts)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
import time
from PIL import Image
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BiRefNet model cache
_birefnet_model = None
_birefnet_transform = None
_device = None

# Processing parameters
SEGMENTATION_THRESHOLD = 0.35  # Lower threshold captures more frizzy edges
MODEL_INPUT_SIZE = (1024, 1024)  # BiRefNet input size
OVERLAY_COLOR = (0, 255, 255)  # Cyan overlay
OVERLAY_ALPHA = 0.5

# Preprocessing settings
ENABLE_PREPROCESSING = True


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


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better segmentation.

    Applies:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
    2. Brightness normalization to maximize dynamic range

    These preprocessing steps improve edge detection and mask quality,
    especially for fine details like hair strands and backlit images.
    """
    if not ENABLE_PREPROCESSING:
        return image

    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge enhanced L channel back
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Brightness normalization to maximize dynamic range
    normalized = cv2.normalize(enhanced_bgr, None, 0, 255, cv2.NORM_MINMAX)

    return normalized


def load_birefnet_model() -> Tuple:
    """
    Load BiRefNet model with caching.

    Returns:
        Tuple of (model, transform) for BiRefNet inference
    """
    global _birefnet_model, _birefnet_transform

    if _birefnet_model is not None and _birefnet_transform is not None:
        logger.info("Using cached BiRefNet model")
        return _birefnet_model, _birefnet_transform

    try:
        from transformers import AutoModelForImageSegmentation

        device = get_device()

        logger.info("Loading BiRefNet_lite model...")
        logger.info("Model: ZhengPeng7/BiRefNet_lite")
        logger.info("Specialization: Salient object detection with fine edge details")

        # Load the model
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_lite",
            trust_remote_code=True
        )
        model = model.to(device)
        model.eval()

        # Create transform for preprocessing
        transform = transforms.Compose([
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Cache the model and transform
        _birefnet_model = model
        _birefnet_transform = transform

        # Get VRAM usage
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"Model loaded - VRAM: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        logger.info("BiRefNet model loaded successfully")
        return model, transform

    except ImportError as e:
        logger.error("transformers package not installed")
        logger.error("Install with: pip install transformers")
        raise e
    except Exception as e:
        logger.error(f"Error loading BiRefNet model: {e}")
        raise e


def create_overlay(image: np.ndarray, mask: np.ndarray, color: tuple = OVERLAY_COLOR, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """Create overlay visualization with semi-transparent mask."""
    overlay = image.copy()

    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    # Blend with original image
    overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)

    # Add boundary for clarity
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


def segment_single_tress(
    image: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    tress_id: int,
    model,
    transform
) -> TressMask:
    """
    Segment a single tress from its cropped region using BiRefNet.

    This function:
    1. Extracts the crop from the full image
    2. Applies preprocessing (CLAHE + normalization) for better segmentation
    3. Runs BiRefNet inference on the crop
    4. Applies threshold and filtering for frizz details
    5. Places mask in full image coordinates

    Args:
        image: Full RGB image
        crop_box: (x, y, w, h) bounding box for this tress
        tress_id: Identifier for this tress
        model: BiRefNet model
        transform: Image preprocessing transform

    Returns:
        TressMask with segmentation results
    """
    x, y, w, h = crop_box

    # Extract crop
    crop = image[y:y+h, x:x+w].copy()
    crop_height, crop_width = crop.shape[:2]

    logger.info(f"Processing tress {tress_id}: crop size {crop_width}x{crop_height} px")

    # Apply preprocessing to enhance image quality
    if ENABLE_PREPROCESSING:
        preprocessed_crop = preprocess_image(crop)
        logger.info("  Applied CLAHE preprocessing for better contrast")
    else:
        preprocessed_crop = crop

    # Convert BGR to RGB for model
    image_rgb = cv2.cvtColor(preprocessed_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Run BiRefNet inference
    try:
        with torch.no_grad():
            # Transform image for model input
            input_tensor = transform(pil_image).unsqueeze(0).cuda()

            # Run model
            result = model(input_tensor)

            # BiRefNet returns a list with one tensor
            if isinstance(result, list) and len(result) > 0:
                mask = result[0]
            else:
                mask = result

            # Convert to numpy array
            if isinstance(mask, torch.Tensor):
                mask = mask.squeeze().cpu().numpy()

            # Apply sigmoid to get probabilities
            mask = 1 / (1 + np.exp(-mask))  # sigmoid

            # Resize mask to match original crop size
            if mask.shape != (crop_height, crop_width):
                mask = cv2.resize(mask, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)

            # Apply threshold for binary mask (lower threshold captures frizzy edges)
            crop_mask = (mask > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255

            # Convert to boolean
            crop_mask = crop_mask.astype(bool)

            # Filter out small isolated regions that are likely background noise
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                crop_mask.astype(np.uint8), connectivity=8
            )

            if num_labels > 1:
                # Keep components that are reasonably large (>300 pixels for substantial hair regions)
                component_areas = stats[1:, cv2.CC_STAT_AREA]
                keep_indices = np.where(component_areas > 300)[0]  # Components > 300 pixels

                if len(keep_indices) > 0:
                    # Create mask with only the kept components
                    filtered_mask = np.zeros(crop_mask.shape, dtype=bool)
                    for idx in keep_indices:
                        label = 1 + idx
                        filtered_mask = filtered_mask | (labels == label)

                    crop_mask = filtered_mask

                    # Apply morphological closing to reconnect nearby hair regions
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    crop_mask = cv2.morphologyEx(crop_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

                    logger.info(f"  Kept {len(keep_indices)} significant components (>300px), filtered {num_labels - 1 - len(keep_indices)} small regions")
                else:
                    # If no components > 300px, keep the largest one
                    largest_label = 1 + np.argmax(component_areas)
                    crop_mask = (labels == largest_label)

                    # Apply morphological closing
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    crop_mask = cv2.morphologyEx(crop_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

                    logger.info(f"  Kept only largest component, filtered {num_labels - 1} small regions")

    except Exception as e:
        logger.error(f"Error during BiRefNet prediction for tress {tress_id}: {e}")
        # Return empty mask as fallback
        crop_mask = np.zeros((crop_height, crop_width), dtype=bool)
        logger.warning(f"  Using empty fallback mask for tress {tress_id}")

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

    logger.info(f"  Segmented {area_pixels:.0f} pixels")

    return TressMask(
        id=tress_id,
        mask=full_mask,
        bbox=crop_box,
        area_pixels=area_pixels,
        center=center,
        confidence=1.0,  # BiRefNet doesn't provide confidence scores like SAM 2
        crop_size=(crop_width, crop_height)
    )


def segment_all_tresses(
    image: np.ndarray,
    tress_boxes: List[Tuple[int, int, int, int]],
    model = None,
    transform = None
) -> SegmentationResult:
    """
    Segment all detected tresses using crop-based BiRefNet processing.

    Processes each tress crop individually, then combines
    the results into a composite mask.

    Args:
        image: Full RGB image
        tress_boxes: List of (x, y, w, h) bounding boxes
        model: BiRefNet model (will load if None)
        transform: Image preprocessing transform (will load if None)

    Returns:
        SegmentationResult with individual tress masks and composite
    """
    start_time = time.time()

    # Load model if not provided
    if model is None or transform is None:
        model, transform = load_birefnet_model()

    device = get_device()

    logger.info(f"\n{'='*70}")
    logger.info(f"SEGMENTING {len(tress_boxes)} TRESSES WITH BIREFNET")
    logger.info(f"{'='*70}")

    # Process each tress
    tress_masks = []

    for i, crop_box in enumerate(tress_boxes, 1):
        logger.info(f"\nTress {i}/{len(tress_boxes)}:")

        tress_mask = segment_single_tress(image, crop_box, i, model, transform)
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
    alpha: float = OVERLAY_ALPHA
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
    for tress in segmentation_result.tresses:
        color = colors[(tress.id - 1) % len(colors)]

        # Create overlay for this tress only
        tress_mask = tress.mask.copy()
        tress_overlay = create_overlay(image, tress_mask, color, alpha)

        # Blend this tress overlay into the main visualization
        mask_indices = tress_mask
        viz_image[mask_indices] = tress_overlay[mask_indices]

        # Draw bounding box
        x, y, w, h = tress.bbox
        cv2.rectangle(viz_image, (x, y), (x + w, y + h), color, 2)

        # Add label
        label = f"T{tress.id}"
        cv2.putText(viz_image, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return viz_image