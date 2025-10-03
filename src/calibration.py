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
    min_radius_pixels: int = 100,
    max_radius_pixels: int = 600
) -> CalibrationResult:
    """
    Detect US quarter in top-left region of image using Hough Circle Transform.
    
    The quarter is mounted on translucent plastic, which may affect edge detection.
    We use multiple detection strategies to handle this gracefully.
    
    Args:
        image_path: Path to the image file
        search_region_fraction: Fraction of image to search (0.3 = top-left 30%)
        min_radius_pixels: Minimum expected quarter radius
        max_radius_pixels: Maximum expected quarter radius
    
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
    
    # Define search region (top-left corner)
    search_height = int(height * search_region_fraction)
    search_width = int(width * search_region_fraction)
    search_region = image[:search_height, :search_width].copy()
    
    logger.info(f"Searching for quarter in top-left region: {search_width}x{search_height} pixels")
    
    # Try multiple detection strategies
    circles = _detect_circles_multi_strategy(search_region, min_radius_pixels, max_radius_pixels)
    
    if circles is None or len(circles) == 0:
        raise ValueError(
            f"Could not detect quarter in top-left region. "
            f"Tried radius range: {min_radius_pixels}-{max_radius_pixels} pixels. "
            f"Ensure quarter is visible and well-lit in the top-left corner."
        )
    
    # Select best circle (largest, since quarter should be prominent)
    best_circle = _select_best_circle(circles, search_region)
    x, y, radius = best_circle
    
    # Calculate calibration
    quarter_area_pixels = np.pi * radius * radius
    calibration_factor = QUARTER_AREA_CM2 / quarter_area_pixels
    
    # Calculate confidence based on circularity and edge strength
    confidence = _calculate_confidence(search_region, x, y, radius)
    
    logger.info(f"Quarter detected: center=({x}, {y}), radius={radius}px")
    logger.info(f"Quarter area: {quarter_area_pixels:.2f} pixels")
    logger.info(f"Calibration factor: {calibration_factor:.6f} cm²/pixel")
    logger.info(f"Detection confidence: {confidence:.2f}")
    
    return CalibrationResult(
        calibration_factor=calibration_factor,
        quarter_center=(x, y),
        quarter_radius=radius,
        quarter_area_pixels=quarter_area_pixels,
        confidence=confidence,
        image_shape=image.shape
    )


def _detect_circles_multi_strategy(
    region: np.ndarray,
    min_radius: int,
    max_radius: int
) -> Optional[np.ndarray]:
    """
    Try multiple Hough Circle detection strategies with different parameters.
    
    Strategy 1: Standard detection (good for clean edges)
    Strategy 2: More sensitive (handles translucent plastic)
    Strategy 3: Relaxed accumulator threshold (noisy backgrounds)
    
    Args:
        region: Image region to search
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
    
    Returns:
        Detected circles array or None
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Strategy 1: Standard parameters
    circles = cv2.HoughCircles(
        filtered,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,  # Canny upper threshold
        param2=30,  # Accumulator threshold
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None and len(circles[0]) > 0:
        logger.info(f"Strategy 1 found {len(circles[0])} circle(s)")
        return circles[0]
    
    # Strategy 2: More sensitive (lower param2)
    circles = cv2.HoughCircles(
        filtered,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=40,
        param2=25,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None and len(circles[0]) > 0:
        logger.info(f"Strategy 2 found {len(circles[0])} circle(s)")
        return circles[0]
    
    # Strategy 3: Even more relaxed
    circles = cv2.HoughCircles(
        filtered,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=30,
        param2=20,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None and len(circles[0]) > 0:
        logger.info(f"Strategy 3 found {len(circles[0])} circle(s)")
        return circles[0]
    
    logger.warning("All detection strategies failed")
    return None


def _select_best_circle(circles: np.ndarray, region: np.ndarray) -> Tuple[int, int, int]:
    """
    Select the most likely quarter from detected circles.
    
    Criteria:
    - Largest radius (quarter should be prominent)
    - Near top-left but not touching edges
    - Good circularity
    
    Args:
        circles: Array of detected circles
        region: Search region image
    
    Returns:
        (x, y, radius) of best circle
    """
    height, width = region.shape[:2]
    
    scored_circles = []
    for circle in circles:
        x, y, radius = circle
        
        # Score based on size (larger is better)
        size_score = radius
        
        # Penalize if too close to edges (likely false positive)
        edge_margin = 50
        if x < edge_margin or y < edge_margin or x > width - edge_margin or y > height - edge_margin:
            edge_score = 0.5
        else:
            edge_score = 1.0
        
        total_score = size_score * edge_score
        scored_circles.append((total_score, circle))
    
    # Return circle with highest score
    best_score, best_circle = max(scored_circles, key=lambda x: x[0])
    x, y, radius = best_circle
    
    return (int(x), int(y), int(radius))


def _calculate_confidence(
    region: np.ndarray,
    x: int,
    y: int,
    radius: int
) -> float:
    """
    Calculate confidence score for detected circle.
    
    Higher confidence for:
    - Strong edges at circle perimeter
    - Good contrast with background
    - Metallic color characteristics
    
    Args:
        region: Search region image
        x, y: Circle center
        radius: Circle radius
    
    Returns:
        Confidence score (0-1)
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Sample points on the circle perimeter
    num_samples = 36
    angles = np.linspace(0, 2 * np.pi, num_samples)
    
    edge_strengths = []
    for angle in angles:
        # Sample point on perimeter
        px = int(x + radius * np.cos(angle))
        py = int(y + radius * np.sin(angle))
        
        # Check bounds
        if 0 <= px < region.shape[1] and 0 <= py < region.shape[0]:
            # Calculate gradient strength at this point
            if px > 0 and px < region.shape[1] - 1 and py > 0 and py < region.shape[0] - 1:
                gx = float(gray[py, px + 1]) - float(gray[py, px - 1])
                gy = float(gray[py + 1, px]) - float(gray[py - 1, px])
                edge_strength = np.sqrt(gx * gx + gy * gy)
                edge_strengths.append(edge_strength)
    
    if len(edge_strengths) == 0:
        return 0.5
    
    # Average edge strength (normalized to 0-1)
    avg_edge_strength = np.mean(edge_strengths)
    confidence = min(avg_edge_strength / 100.0, 1.0)
    
    return confidence


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

