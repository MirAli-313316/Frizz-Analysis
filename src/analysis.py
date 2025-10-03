"""
Analysis module combining calibration and segmentation for surface area calculations.

This module integrates the calibration and segmentation systems to provide
complete hair tress analysis with quantitative surface area measurements in cm².
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict

from .calibration import detect_quarter
from .segmentation import segment_tresses, SegmentationResult, TressMask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TressAnalysis:
    """Container for individual tress analysis results."""
    
    tress_id: int
    area_cm2: float  # Surface area in cm²
    pixel_count: int  # Total pixels in mask
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]  # (x, y) center position
    aspect_ratio: float  # height/width
    confidence: float  # SAM prediction confidence
    
    def __repr__(self):
        return (f"TressAnalysis(id={self.tress_id}, area={self.area_cm2:.2f}cm², "
                f"pixels={self.pixel_count}, confidence={self.confidence:.2f})")


@dataclass
class ImageAnalysis:
    """Container for complete image analysis results."""
    
    image_path: str
    image_name: str
    tresses: List[TressAnalysis]
    calibration_factor: float  # cm² per pixel
    quarter_info: Dict  # Quarter detection details
    processing_time: float
    device_used: str
    
    def get_total_area(self) -> float:
        """Calculate total surface area across all tresses."""
        return sum(t.area_cm2 for t in self.tresses)
    
    def get_tress_by_id(self, tress_id: int) -> Optional[TressAnalysis]:
        """Get specific tress by ID."""
        for tress in self.tresses:
            if tress.tress_id == tress_id:
                return tress
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'image_path': self.image_path,
            'image_name': self.image_name,
            'tresses': [asdict(t) for t in self.tresses],
            'calibration_factor': self.calibration_factor,
            'quarter_info': self.quarter_info,
            'processing_time': self.processing_time,
            'device_used': self.device_used,
            'total_area_cm2': self.get_total_area(),
            'tress_count': len(self.tresses)
        }
    
    def __repr__(self):
        return (f"ImageAnalysis({self.image_name}, {len(self.tresses)} tresses, "
                f"total={self.get_total_area():.2f}cm²)")


def analyze_image(
    image_path: str,
    visualize: bool = False,
    output_dir: Optional[str] = None,
    max_processing_dim: int = 1024
) -> ImageAnalysis:
    """
    Complete analysis of hair tress image: calibration + segmentation + area calculation.
    
    Processing steps:
    1. Detect quarter and calculate calibration factor (cm²/pixel)
    2. Run SAM segmentation to detect tress masks
    3. Calculate surface area for each tress (pixels × calibration_factor)
    4. Optionally generate visualization overlays
    
    Args:
        image_path: Path to image file
        visualize: Whether to generate visualization images
        output_dir: Directory for visualization outputs (default: outputs/)
        max_processing_dim: Max dimension for SAM processing (memory optimization)
    
    Returns:
        ImageAnalysis object with complete results
    
    Raises:
        ValueError: If quarter not detected or image invalid
        FileNotFoundError: If image file doesn't exist
    """
    import time
    start_time = time.time()
    
    # Validate input
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image_name = image_path.stem
    logger.info(f"=" * 70)
    logger.info(f"Analyzing: {image_name}")
    logger.info(f"=" * 70)
    
    # Step 1: Calibration - detect quarter and get cm²/pixel factor
    logger.info("Step 1: Calibrating with quarter...")
    try:
        quarter_result = detect_quarter(str(image_path))
        
        calibration_factor = quarter_result.calibration_factor
        
        logger.info(f"✓ Quarter detected: center={quarter_result.quarter_center}, "
                   f"radius={quarter_result.quarter_radius:.1f}px")
        logger.info(f"✓ Calibration factor: {calibration_factor:.6f} cm²/pixel")
        
        # Create exclude region around quarter for segmentation
        qx, qy = quarter_result.quarter_center
        qr = int(quarter_result.quarter_radius)
        margin = 50
        exclude_region = (
            max(0, qx - qr - margin),
            max(0, qy - qr - margin),
            (qr + margin) * 2,
            (qr + margin) * 2
        )
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise
    
    # Step 2: Segmentation - detect tress masks
    logger.info("Step 2: Segmenting hair tresses...")
    try:
        seg_result = segment_tresses(
            str(image_path),
            exclude_region=exclude_region,
            min_tress_area=200000,  # ~10 cm²
            max_tress_area=1800000,  # ~90 cm²
            brightness_threshold=150,
            max_processing_dim=max_processing_dim
        )
        
        logger.info(f"✓ Detected {len(seg_result.tresses)} tresses")
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise
    
    # Step 3: Calculate surface areas
    logger.info("Step 3: Calculating surface areas...")
    tress_analyses = []
    
    for tress_mask in seg_result.tresses:
        # Calculate area in cm² = pixel_count × calibration_factor
        area_cm2 = tress_mask.area_pixels * calibration_factor
        
        tress_analysis = TressAnalysis(
            tress_id=tress_mask.id,
            area_cm2=area_cm2,
            pixel_count=int(tress_mask.area_pixels),
            bbox=tress_mask.bbox,
            center=tress_mask.center,
            aspect_ratio=tress_mask.aspect_ratio,
            confidence=tress_mask.confidence
        )
        
        tress_analyses.append(tress_analysis)
        logger.info(f"  Tress {tress_mask.id}: {area_cm2:.2f} cm² "
                   f"({int(tress_mask.area_pixels)} pixels)")
    
    total_area = sum(t.area_cm2 for t in tress_analyses)
    logger.info(f"✓ Total surface area: {total_area:.2f} cm²")
    
    # Step 4: Optional visualization
    if visualize:
        logger.info("Step 4: Generating visualizations...")
        if output_dir is None:
            output_dir = Path("outputs")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        _create_analysis_visualization(
            str(image_path),
            seg_result,
        tress_analyses,
        {
            'center': quarter_result.quarter_center,
            'radius': quarter_result.quarter_radius
        },
        output_dir / f"analysis_{image_name}.jpg"
        )
        logger.info(f"✓ Saved visualization to outputs/")
    
    processing_time = time.time() - start_time
    logger.info(f"✓ Analysis complete in {processing_time:.2f}s")
    logger.info(f"=" * 70)
    
    # Create result object
    result = ImageAnalysis(
        image_path=str(image_path),
        image_name=image_name,
        tresses=tress_analyses,
        calibration_factor=calibration_factor,
        quarter_info={
            'center': quarter_result.quarter_center,
            'radius': quarter_result.quarter_radius,
            'confidence': quarter_result.confidence
        },
        processing_time=processing_time,
        device_used=seg_result.device_used
    )
    
    return result


def _create_analysis_visualization(
    image_path: str,
    seg_result: SegmentationResult,
    tress_analyses: List[TressAnalysis],
    quarter_result: Dict,
    output_path: Path
) -> None:
    """
    Create visualization showing quarter, tresses, and surface areas.
    
    Args:
        image_path: Path to original image
        seg_result: Segmentation results
        tress_analyses: List of tress analysis results
        quarter_result: Quarter detection results
        output_path: Where to save visualization
    """
    # Load original image
    image = cv2.imread(image_path)
    vis_image = image.copy()
    
    # Draw quarter
    qx, qy = quarter_result['center']
    qr = int(quarter_result['radius'])
    cv2.circle(vis_image, (qx, qy), qr, (0, 255, 255), 3)
    cv2.circle(vis_image, (qx, qy), 5, (0, 255, 255), -1)
    
    # Generate colors for tresses
    colors = _generate_distinct_colors(len(seg_result.tresses))
    
    # Draw tresses with area labels
    overlay = image.copy()
    
    for tress_mask, tress_analysis, color in zip(
        seg_result.tresses, tress_analyses, colors
    ):
        # Apply colored mask
        overlay[tress_mask.mask] = color
        
        # Draw bounding box
        x, y, w, h = tress_mask.bbox
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 3)
        
        # Add label with area
        label = f"#{tress_mask.id}: {tress_analysis.area_cm2:.1f} cm²"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Draw background
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
    
    # Blend overlay
    alpha = 0.3
    vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)
    
    # Add summary
    total_area = sum(t.area_cm2 for t in tress_analyses)
    summary_lines = [
        f"Tresses: {len(tress_analyses)}",
        f"Total Area: {total_area:.2f} cm²",
        f"Avg Area: {total_area/len(tress_analyses):.2f} cm²"
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    y_offset = 50
    
    for i, text in enumerate(summary_lines):
        y_pos = y_offset + i * 50
        
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
    
    # Save
    cv2.imwrite(str(output_path), vis_image)


def _generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr)))
    return colors


def batch_analyze_images(
    image_paths: List[str],
    visualize: bool = False,
    output_dir: Optional[str] = None,
    max_processing_dim: int = 1024
) -> List[ImageAnalysis]:
    """
    Analyze multiple images in batch.
    
    Args:
        image_paths: List of image file paths
        visualize: Whether to generate visualizations
        output_dir: Directory for outputs
        max_processing_dim: Max dimension for SAM processing
    
    Returns:
        List of ImageAnalysis results
    """
    results = []
    
    logger.info(f"\nBatch processing {len(image_paths)} images...")
    logger.info("=" * 70)
    
    for i, image_path in enumerate(image_paths, 1):
        logger.info(f"\nImage {i}/{len(image_paths)}")
        try:
            result = analyze_image(
                image_path,
                visualize=visualize,
                output_dir=output_dir,
                max_processing_dim=max_processing_dim
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to analyze {image_path}: {e}")
            logger.error("Continuing with next image...")
            continue
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Batch processing complete: {len(results)}/{len(image_paths)} successful")
    logger.info(f"{'=' * 70}\n")
    
    return results

