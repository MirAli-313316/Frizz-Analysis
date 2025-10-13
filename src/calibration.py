"""
Calibration module for detecting US quarter and calculating pixel-to-cm² conversion.

Scientific Constants:
- US Quarter diameter: 24.26 mm = 2.426 cm
- US Quarter area: π × (1.213)² = 4.62 cm²

This module detects the quarter in each image for per-image calibration,
accounting for potential camera distance variations between shots.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quarter physical properties
QUARTER_DIAMETER_CM = 2.426
QUARTER_AREA_CM2 = 4.62


def _detect_coin_sam2_optimized_quarter(region_bgr: np.ndarray,
                                        expected_tresses: int) -> Optional[Tuple[int, int, int]]:
    """
    Detect US quarter using hybrid approach: Hough circles for detection, SAM 2 for segmentation.

    This approach leverages the strengths of both methods:
    1. Hough Circle detection reliably finds circular objects
    2. SAM 2 provides precise segmentation boundaries

    Args:
        region_bgr: Input region in BGR format
        expected_tresses: Expected number of tresses (for compatibility)

    Returns (cx, cy, r) in region coordinates. No resizing is performed.
    """
    try:
        import torch
        from .segmentation import load_sam2_model
    except Exception:
        return None

    # Convert to RGB for SAM 2
    region_rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    h, w = region_rgb.shape[:2]

    logger.info(f"Detecting quarter in {w}x{h} region using hybrid approach")

    # Step 1: Find quarter candidates using Hough Circle detection
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=50,
        maxRadius=400
    )

    if circles is None:
        logger.warning("No circles detected by Hough Circle Transform")
        return None

    logger.info(f"Hough Circle detection found {len(circles[0])} candidates")

    # Step 2: Use SAM 2 for precise segmentation of detected circles
    circles = np.round(circles[0, :]).astype("int")

    # Sort by radius (largest first) - quarters should be among the larger circles
    circles = sorted(circles, key=lambda c: c[2], reverse=True)

    best_result = None

    for i, circle in enumerate(circles[:5]):  # Try top 5 largest circles
        x, y, r = circle
        logger.info(f"  Testing circle {i+1}: center=({x}, {y}), radius={r}")

        # Use single point at circle center for SAM 2 segmentation
        input_points = np.array([[x, y]], dtype=np.float32)
        input_labels = np.array([1], dtype=np.int32)

        try:
            predictor = load_sam2_model()  # Use same model as segmentation (Large)
            with torch.inference_mode():
                predictor.set_image(region_rgb)
                masks, scores, logits = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )

            if len(masks) == 0:
                logger.warning(f"  No masks generated for circle at ({x}, {y})")
                continue

            # Use the highest scoring mask
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]

            # Calculate mask properties
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.warning(f"  No contours found in mask for circle at ({x}, {y})")
                continue

            # Use the largest contour from the mask
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area < 500:  # Minimum reasonable area for a quarter
                logger.warning(f"  Mask area too small ({area}px) for circle at ({x}, {y})")
                continue

            per = cv2.arcLength(largest_contour, True)
            if per <= 1:
                continue

            circularity = 4.0 * np.pi * area / (per * per)

            # Recalculate center and radius from the segmented contour
            (cx, cy), rr = cv2.minEnclosingCircle(largest_contour)
            new_r = int(rr)

            # Score based on circularity and size match with original detection
            size_match = 1.0 - abs(new_r - r) / max(r, new_r)  # How well radius matches
            combined_score = (circularity * 0.7) + (size_match * 0.3)

            logger.info(f"  Circle at ({x}, {y}): mask_circularity={circularity:.3f}, "
                       f"size_match={size_match:.3f}, combined_score={combined_score:.3f}, "
                       f"final_radius={new_r}")

            if best_result is None or combined_score > best_result[0]:
                best_result = (combined_score, (int(cx), int(cy), new_r))

        except Exception as e:
            logger.warning(f"SAM 2 segmentation failed for circle at ({x}, {y}): {e}")
            continue

    if best_result is not None:
        score, (cx, cy, r) = best_result
        logger.info(f"Selected best quarter: score={score:.3f}, center=({cx}, {cy}), radius={r}")
        return (cx, cy, r)
    else:
        logger.warning("No suitable quarter candidate found by hybrid approach")
        return None


class CalibrationResult:
    """Container for calibration results and metadata."""
    
    def __init__(
        self,
        calibration_factor: float,
        quarter_center: Tuple[int, int],
        quarter_radius: int,
        quarter_area_pixels: float,
        confidence: float,
        image_shape: Tuple[int, int, int]
    ):
        """
        Args:
            calibration_factor: cm² per pixel conversion factor
            quarter_center: (x, y) center coordinates in pixels
            quarter_radius: Detected radius in pixels
            quarter_area_pixels: Quarter area in pixels
            confidence: Detection confidence score (0-1)
            image_shape: Original image shape (height, width, channels)
        """
        self.calibration_factor = calibration_factor
        self.quarter_center = quarter_center
        self.quarter_radius = quarter_radius
        self.quarter_area_pixels = quarter_area_pixels
        self.confidence = confidence
        self.image_shape = image_shape
    
    def __repr__(self):
        return (f"CalibrationResult(factor={self.calibration_factor:.6f} cm²/px, "
                f"center={self.quarter_center}, radius={self.quarter_radius}px, "
                f"confidence={self.confidence:.2f})")


def detect_quarter(
    image_path: str,
    search_region_fraction: float = 0.3,
    expected_tresses: int = 6
) -> CalibrationResult:
    """
    Detect US quarter in top-left region of image using SAM 2 Large (same as segmentation).

    Uses hybrid Hough Circle + SAM 2 approach for reliable quarter detection.
    Uses the same SAM 2 Large model as the segmentation process for consistency.

    Args:
        image_path: Path to the image file
        search_region_fraction: Fraction of image to search (0.3 = top-left 30%)
        expected_tresses: Expected number of tresses (for compatibility)

    Returns:
        CalibrationResult containing calibration factor and detection metadata

    Raises:
        ValueError: If quarter cannot be detected reliably
    """
    logger.info(f"Loading image: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]
    logger.info(f"Image dimensions: {width}x{height} pixels")

    # Use SAM 2 Large for quarter detection (same model as segmentation)
    roi_h = min(700, height)
    roi_w = min(700, width)
    roi = image[:roi_h, :roi_w].copy()

    logger.info("Detecting quarter with SAM 2 (consistent with segmentation model)...")
    sam2_coin = _detect_coin_sam2_optimized_quarter(roi, expected_tresses=expected_tresses)
    if sam2_coin is not None:
        rx, ry, rr = sam2_coin
        quarter_area_pixels = np.pi * rr * rr
        calibration_factor = QUARTER_AREA_CM2 / quarter_area_pixels
        confidence = _calculate_confidence_simple(roi, rx, ry, rr)
        logger.info(f"SAM2 quarter: center=({rx}, {ry}), radius={rr}px (fixed 700x700 ROI)")
        return CalibrationResult(
            calibration_factor=calibration_factor,
            quarter_center=(rx, ry),
            quarter_radius=rr,
            quarter_area_pixels=quarter_area_pixels,
            confidence=confidence,
            image_shape=image.shape
        )

    # SAM 2 detection failed
    raise ValueError(
        "SAM 2 coin detection failed on the fixed 700x700 top-left crop. "
        "Try brightening or shifting the coin slightly, ensure it is fully within the top-left 700px."
    )


def _calculate_confidence_simple(
    region: np.ndarray,
    x: int,
    y: int,
    radius: int
) -> float:
    """
    Simple confidence calculation for SAM 2 detected circle.

    Args:
        region: Search region image
        x, y: Circle center
        radius: Circle radius

    Returns:
        Confidence score (0-1)
    """
    # Basic validation - if we got a reasonable radius, give high confidence
    if 50 <= radius <= 300:
        return 0.9
    elif 30 <= radius <= 400:
        return 0.7
    else:
        return 0.5


def create_visualization(
    image_path: str,
    calibration_result: CalibrationResult,
    show_measurements: bool = True
) -> np.ndarray:
    """
    Create visualization showing detected quarter with overlay.
    
    Args:
        image_path: Path to original image
        calibration_result: Calibration result to visualize
        show_measurements: Whether to show measurement annotations
    
    Returns:
        Image array with visualization overlay
    """
    # Load original image
    image = cv2.imread(image_path)
    vis_image = image.copy()
    
    x, y = calibration_result.quarter_center
    radius = calibration_result.quarter_radius
    
    # Draw circle outline (green)
    cv2.circle(vis_image, (x, y), radius, (0, 255, 0), 3)
    
    # Draw center point (red)
    cv2.circle(vis_image, (x, y), 5, (0, 0, 255), -1)
    
    # Draw crosshair
    cv2.line(vis_image, (x - 20, y), (x + 20, y), (0, 0, 255), 2)
    cv2.line(vis_image, (x, y - 20), (x, y + 20), (0, 0, 255), 2)
    
    if show_measurements:
        # Add text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        color = (0, 255, 0)
        bg_color = (0, 0, 0)
        
        # Background rectangles for better text visibility
        texts = [
            f"Quarter Detected",
            f"Radius: {radius}px",
            f"Area: {calibration_result.quarter_area_pixels:.0f}px",
            f"Factor: {calibration_result.calibration_factor:.6f} cm²/px",
            f"Confidence: {calibration_result.confidence:.0%}"
        ]
        
        text_x = x + radius + 20
        text_y_start = y - 100
        
        for i, text in enumerate(texts):
            text_y = text_y_start + i * 50
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(
                vis_image,
                (text_x - 5, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + baseline + 5),
                bg_color,
                -1
            )
            
            # Draw text
            cv2.putText(vis_image, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return vis_image


def pixels_to_cm2(pixel_area: float, calibration_factor: float) -> float:
    """
    Convert pixel area to cm² using calibration factor.
    
    Args:
        pixel_area: Area in pixels
        calibration_factor: Calibration factor (cm²/pixel)
    
    Returns:
        Area in cm²
    """
    return pixel_area * calibration_factor


def cm2_to_pixels(area_cm2: float, calibration_factor: float) -> float:
    """
    Convert cm² to pixel area using calibration factor.
    
    Args:
        area_cm2: Area in cm²
        calibration_factor: Calibration factor (cm²/pixel)
    
    Returns:
        Area in pixels
    """
    return area_cm2 / calibration_factor

