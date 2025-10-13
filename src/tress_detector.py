"""
Tress detection module using OpenCV for fast, lightweight detection.

This module uses traditional computer vision techniques to quickly identify
individual hair tress locations before high-resolution SAM 2 segmentation.
Detection is based on dark regions (hair) against white background.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hard-coded detection parameters optimized for our use case
BINARY_THRESHOLD = 200  # Hair is dark (<200), background is white (>200)
MIN_TRESS_AREA = 50000  # Minimum pixels for valid tress (filters noise)
BBOX_PADDING = 100  # Padding around bounding box to capture frizz
MORPH_KERNEL_SIZE = 15  # Kernel size for morphological closing
MAX_ASPECT_RATIO = 8.0  # Filter out extremely wide strips (likely background/frame)
MAX_EDGE_TOUCH_FRACTION = 0.98  # Filter boxes that cover entire width/height


def detect_tress_regions(
    image: np.ndarray,
    calibration_box: Optional[Tuple[int, int, int, int]] = None,
    num_expected_tresses: int = 7
) -> List[Tuple[int, int, int, int]]:
    """
    Detect individual hair tress locations using OpenCV.
    
    This function quickly identifies dark regions (hair tresses) against
    a white background using binary thresholding and morphological operations.
    
    Args:
        image: RGB image array (height, width, 3)
        calibration_box: Optional (x, y, w, h) region to exclude (quarter)
        num_expected_tresses: Expected number of tresses (for validation only)
    
    Returns:
        List of bounding boxes [(x, y, w, h), ...] sorted by position (top to bottom, left to right)
    
    Algorithm:
        1. Convert to grayscale
        2. Binary threshold (hair=white, background=black after THRESH_BINARY_INV)
        3. Mask out calibration quarter region
        4. Morphological closing to connect hair strands
        5. Find external contours
        6. Filter by minimum area
        7. Get bounding rectangles with padding
        8. Sort by position (y, then x)
    """
    logger.info("Detecting tress regions using OpenCV...")
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Step 2: Binary threshold - invert so hair is white on black background
    _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # Step 3: Mask out calibration quarter if provided
    if calibration_box is not None:
        x, y, w, h = calibration_box
        # Expand mask slightly to ensure complete exclusion
        mask_expansion = 20
        x = max(0, x - mask_expansion)
        y = max(0, y - mask_expansion)
        w = w + 2 * mask_expansion
        h = h + 2 * mask_expansion
        
        # Set quarter region to black (exclude from detection)
        binary[y:y+h, x:x+w] = 0
        logger.info(f"Excluded calibration region: ({x}, {y}, {w}, {h})")
    
    # Step 4: Morphological closing to connect nearby hair strands
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Step 5: Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    logger.info(f"Found {len(contours)} contours")
    
    # Step 6: Filter contours by area and shape to avoid full-image/edges
    valid_contours = []
    img_height, img_width = image.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_TRESS_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Reject near-full-image regions (likely background merge)
        if w >= int(img_width * MAX_EDGE_TOUCH_FRACTION) and h >= int(img_height * 0.3):
            continue
        if h >= int(img_height * MAX_EDGE_TOUCH_FRACTION) and w >= int(img_width * 0.3):
            continue

        # Reject extremely wide/flat regions (frame/background)
        aspect = max(w / max(1, h), h / max(1, w))
        if aspect > MAX_ASPECT_RATIO:
            continue

        # Reject boxes that touch both left and right or both top and bottom edges
        touches_left = x <= 2
        touches_right = x + w >= img_width - 2
        touches_top = y <= 2
        touches_bottom = y + h >= img_height - 2
        if (touches_left and touches_right) or (touches_top and touches_bottom):
            continue

        valid_contours.append(contour)
    
    logger.info(f"Filtered to {len(valid_contours)} valid tresses (area >= {MIN_TRESS_AREA} pixels)")
    
    # Step 7: Get bounding rectangles with padding
    bounding_boxes = []
    
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding to capture frizz extending beyond main body
        x_padded = max(0, x - BBOX_PADDING)
        y_padded = max(0, y - BBOX_PADDING)
        w_padded = min(img_width - x_padded, w + 2 * BBOX_PADDING)
        h_padded = min(img_height - y_padded, h + 2 * BBOX_PADDING)
        
        bounding_boxes.append((x_padded, y_padded, w_padded, h_padded))
    
    # Step 8: Sort by position (top to bottom, then left to right)
    # This ensures consistent tress numbering across images
    bounding_boxes.sort(key=lambda box: (box[1], box[0]))  # Sort by (y, x)
    
    # Log detection results
    logger.info(f"Detected {len(bounding_boxes)} tress regions:")
    for i, (x, y, w, h) in enumerate(bounding_boxes, 1):
        logger.info(f"  Tress {i}: position=({x}, {y}), size=({w}x{h}) pixels")
    
    # Validation warning
    if len(bounding_boxes) != num_expected_tresses:
        logger.warning(f"Expected {num_expected_tresses} tresses but detected {len(bounding_boxes)}")
        logger.warning("This may be normal if the actual count differs from expected")
    
    return bounding_boxes


def visualize_tress_detection(
    image: np.ndarray,
    bounding_boxes: List[Tuple[int, int, int, int]],
    calibration_box: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """
    Create visualization of detected tress regions.
    
    Draws labeled rectangles around each detected tress for visual verification.
    
    Args:
        image: RGB image array
        bounding_boxes: List of (x, y, w, h) bounding boxes
        calibration_box: Optional (x, y, w, h) for quarter region
    
    Returns:
        Annotated image with rectangles and labels
    """
    # Create copy to avoid modifying original
    viz_image = image.copy()
    
    # Draw calibration box in blue if provided
    if calibration_box is not None:
        x, y, w, h = calibration_box
        cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(viz_image, "Quarter", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Draw each tress bounding box in green
    for i, (x, y, w, h) in enumerate(bounding_boxes, 1):
        # Draw rectangle
        cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Add label
        label = f"Tress {i}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        
        # Position label above box (or inside if at top of image)
        if y > 40:
            label_y = y - 10
        else:
            label_y = y + 35
        
        # Draw label background for readability
        cv2.rectangle(viz_image,
                     (x, label_y - label_size[1] - 5),
                     (x + label_size[0] + 5, label_y + 5),
                     (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(viz_image, label, (x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # Add dimensions text inside box
        dim_text = f"{w}x{h}px"
        cv2.putText(viz_image, dim_text, (x + 10, y + h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return viz_image


def get_detection_stats(bounding_boxes: List[Tuple[int, int, int, int]]) -> dict:
    """
    Calculate statistics about detected tresses.
    
    Args:
        bounding_boxes: List of (x, y, w, h) bounding boxes
    
    Returns:
        Dictionary with detection statistics
    """
    if not bounding_boxes:
        return {
            'count': 0,
            'mean_width': 0,
            'mean_height': 0,
            'mean_area': 0,
            'min_area': 0,
            'max_area': 0
        }
    
    widths = [w for _, _, w, _ in bounding_boxes]
    heights = [h for _, _, _, h in bounding_boxes]
    areas = [w * h for _, _, w, h in bounding_boxes]
    
    return {
        'count': len(bounding_boxes),
        'mean_width': np.mean(widths),
        'mean_height': np.mean(heights),
        'mean_area': np.mean(areas),
        'min_area': np.min(areas),
        'max_area': np.max(areas),
        'total_area': np.sum(areas)
    }
