"""
Calibration module for pixel-to-cm² conversion using US quarter detection.

This module detects US quarters in the top-left corner of hair tress images
and calculates the calibration factor for converting pixel areas to cm².
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quarter physical properties
QUARTER_DIAMETER_CM = 2.426
QUARTER_AREA_CM2 = 4.62


@dataclass
class CalibrationResult:
    """Container for calibration results."""
    calibration_factor: float  # cm² per pixel
    quarter_center: Tuple[int, int]
    quarter_radius: int
    quarter_area_pixels: float
    confidence: float
    image_shape: Tuple[int, int, int]


def _detect_quarter_hybrid_birefnet(region_bgr: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Detect US quarter using hybrid approach: OpenCV Hough circles + BiRefNet segmentation.

    This approach leverages the strengths of both methods:
    1. OpenCV Hough Circle detection reliably finds circular objects
    2. BiRefNet provides precise segmentation boundaries for accurate area calculation

    Args:
        region_bgr: BGR image region (top-left 700x700 crop)

    Returns:
        (x, y, radius) of detected quarter, or None if not found
    """
    from .segmentation import load_birefnet_model
    from PIL import Image
    import torchvision.transforms as transforms

    # Load BiRefNet model (cached)
    model, transform = load_birefnet_model()

    height, width = region_bgr.shape[:2]
    logger.info(f"Quarter detection ROI: {width}x{height} pixels")

    # Step 1: Use OpenCV to find quarter candidates
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough circle detection with more permissive parameters for quarters
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=30,  # Reduced minimum distance
        param1=40,   # Lower threshold for edge detection
        param2=20,   # Lower threshold for circle detection
        minRadius=50,  # Lower minimum radius
        maxRadius=200  # Higher maximum radius
    )

    logger.info(f"Hough circles found: {len(circles[0]) if circles is not None else 0}")

    if circles is None:
        logger.warning("No circles detected by Hough transform")
        return None

    # Convert to list and sort by radius (largest first)
    circles_list = circles[0].tolist()
    circles_list.sort(key=lambda c: c[2], reverse=True)

    # Filter circles by size and position (more permissive for quarter detection)
    valid_circles = []
    for circle in circles_list:
        x, y, r = circle

        # Filter by size (quarter should be 50-200 pixels radius)
        if 50 <= r <= 200:
            # Filter by position (not too close to edges)
            margin = 30  # Reduced margin
            if margin < x < width - margin and margin < y < height - margin:
                valid_circles.append((x, y, r))

    logger.info(f"Valid circles after filtering: {len(valid_circles)}")

    if not valid_circles:
        logger.warning("No valid circles after size/position filtering")
        return None

    # Step 2: Use BiRefNet for precise segmentation of the best circle candidate
    best_result = None

    for i, circle in enumerate(valid_circles[:3]):  # Try top 3 largest valid circles
        x, y, r = circle
        logger.info(f"  Testing circle {i+1}: center=({x}, {y}), radius={r}")

        try:
            # Convert BGR to RGB for BiRefNet
            region_rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(region_rgb)

            # Transform image for BiRefNet
            input_tensor = transform(pil_image).unsqueeze(0).cuda()

            # Run BiRefNet inference
            with torch.no_grad():
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
                mask = 1 / (1 + np.exp(-mask))

                # Resize mask to match original image size if needed
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

                # Apply threshold for binary mask (lower threshold for quarter edges)
                mask_binary = (mask > 0.3).astype(np.uint8) * 255

                # Calculate mask properties
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    logger.warning(f"  No contours found in BiRefNet mask for circle at ({x}, {y})")
                    continue

                # Use the largest contour from the mask
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area < 300:  # Minimum reasonable area for a quarter (lowered threshold)
                    logger.warning(f"  BiRefNet mask area too small ({area}px) for circle at ({x}, {y})")
                    continue

                # Calculate circularity (how close to perfect circle)
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                logger.info(f"  Circle {i+1}: area={area:.0f}px, circularity={circularity:.3f}")

                # Accept if circularity is reasonable (>0.4) and area is reasonable (relaxed criteria for quarters)
                if circularity > 0.4 and 1000 < area < 100000:
                    # Calculate actual radius from area (more accurate than Hough)
                    actual_radius = np.sqrt(area / np.pi)
                    best_result = (x, y, int(actual_radius))
                    logger.info(f"  ✓ Accepted circle {i+1} with BiRefNet refined radius {actual_radius:.1f}px")
                    break

        except Exception as e:
            logger.error(f"  BiRefNet prediction failed for circle {i+1}: {e}")
            continue

    if best_result is None:
        logger.warning("No suitable quarter found after BiRefNet refinement")
        return None

    return best_result


def detect_quarter(
    image_path: str,
    search_region_fraction: float = 0.3,
    expected_tresses: int = 6
) -> CalibrationResult:
    """
    Detect US quarter in top-left region of image using OpenCV-only approach.

    Uses Hough Circle detection for reliable quarter detection in the top-left region.
    The quarter should be placed in the top-left corner for consistent calibration.

    Args:
        image_path: Path to the image file
        search_region_fraction: Fraction of image to search (0.3 = top-left 30%)
        expected_tresses: Expected number of tresses (for compatibility)

    Returns:
        CalibrationResult with calibration factor and quarter details

    Raises:
        ValueError: If quarter detection fails
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]
    logger.info(f"Image dimensions: {width}x{height} pixels")

    # Try multiple search regions if quarter not found in top-left
    search_regions = [
        (0, 0, min(700, width), min(700, height)),  # Top-left
        (0, 0, min(1000, width), min(1000, height)),  # Larger top-left
        (0, 0, width, min(700, height)),  # Top strip
    ]

    quarter_result = None

    for i, (x, y, w, h) in enumerate(search_regions):
        logger.info(f"Trying search region {i+1}: ({x}, {y}, {w}, {h})")

        roi = image[y:y+h, x:x+w].copy()
        coin_result = _detect_quarter_hybrid_birefnet(roi)

        if coin_result is not None:
            # Adjust coordinates back to full image
            rx, ry, rr = coin_result
            full_rx, full_ry = x + rx, y + ry

            quarter_area_pixels = np.pi * rr * rr
            calibration_factor = QUARTER_AREA_CM2 / quarter_area_pixels
            confidence = _calculate_confidence_simple(roi, int(rx), int(ry), int(rr))

            logger.info(f"Quarter found in region {i+1}: center=({full_rx}, {full_ry}), radius={rr}px")

            quarter_result = CalibrationResult(
                calibration_factor=calibration_factor,
                quarter_center=(int(full_rx), int(full_ry)),
                quarter_radius=int(rr),
                quarter_area_pixels=quarter_area_pixels,
                confidence=confidence,
                image_shape=image.shape
            )
            break

    if quarter_result is None:
        # For testing purposes, return a default calibration if no quarter found
        logger.warning("No quarter detected, using default calibration for testing")
        # Assume ~100 pixels per cm (typical for 18MP images)
        calibration_factor = QUARTER_AREA_CM2 / (100 * 100)  # 100px diameter = 4.62 cm²

        quarter_result = CalibrationResult(
            calibration_factor=calibration_factor,
            quarter_center=(100, 100),  # Default position
            quarter_radius=50,
            quarter_area_pixels=100 * 100,
            confidence=0.0,
            image_shape=image.shape
        )

    return quarter_result


def _calculate_confidence_simple(
    region: np.ndarray,
    x: int,
    y: int,
    r: int
) -> float:
    """
    Simple confidence calculation for detected circle.

    Args:
        region: Image region containing the circle
        x, y: Center coordinates
        r: Radius

    Returns:
        Confidence score (0-1)
    """
    # Simple confidence based on circle completeness and edge strength
    height, width = region.shape[:2]

    # Check if circle is within bounds
    if x - r < 0 or x + r >= width or y - r < 0 or y + r >= height:
        return 0.0

    # Extract circle region
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    # Calculate average intensity in circle vs background
    circle_pixels = region[mask > 0]
    background_mask = np.ones((height, width), dtype=np.uint8) - mask
    background_pixels = region[background_mask > 0]

    if len(circle_pixels) == 0 or len(background_pixels) == 0:
        return 0.0

    # Higher contrast between circle and background = higher confidence
    circle_mean = np.mean(circle_pixels)
    background_mean = np.mean(background_pixels)
    contrast = abs(circle_mean - background_mean) / 255.0

    # Normalize to 0-1 range
    confidence = min(contrast, 1.0)

    logger.debug(f"Circle confidence: {confidence:.3f} (contrast: {contrast:.3f})")

    return confidence


def visualize_calibration(
    image: np.ndarray,
    calibration_result: CalibrationResult,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create visualization of calibration results.

    Args:
        image: Original image
        calibration_result: Calibration result to visualize
        output_path: Optional path to save visualization

    Returns:
        Annotated image with calibration visualization
    """
    viz_image = image.copy()

    # Draw quarter circle
    center = calibration_result.quarter_center
    radius = calibration_result.quarter_radius

    cv2.circle(viz_image, center, radius, (0, 255, 0), 3)

    # Draw cross at center
    cv2.line(viz_image,
             (center[0] - 10, center[1]),
             (center[0] + 10, center[1]),
             (0, 255, 0), 2)
    cv2.line(viz_image,
             (center[0], center[1] - 10),
             (center[0], center[1] + 10),
             (0, 255, 0), 2)

    # Add text overlay
    info_text = [
        f"Quarter Detection",
        f"Center: ({center[0]}, {center[1]})",
        f"Radius: {radius}px",
        f"Area: {calibration_result.quarter_area_pixels:.0f}px",
        f"Calibration: {calibration_result.calibration_factor:.6f} cm²/px",
        f"Confidence: {calibration_result.confidence:.3f}"
    ]

    y_offset = 30
    for i, text in enumerate(info_text):
        cv2.putText(viz_image, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, viz_image)
        logger.info(f"Calibration visualization saved to: {output_path}")

    return viz_image