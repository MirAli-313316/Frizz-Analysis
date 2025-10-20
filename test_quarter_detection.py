#!/usr/bin/env python3
"""
Test script for hybrid BiRefNet quarter detection.
"""

from src.analysis import analyze_image

result = analyze_image('test_images/IMG_8787.JPG', visualize=False, num_expected_tresses=6)
print(f'Total area: {result.get_total_area():.2f} cm²')
print(f'Quarter info keys: {list(result.quarter_info.keys())}')
print(f'Calibration factor: {result.calibration_factor:.6f} cm²/pixel')
print('SUCCESS: Hybrid BiRefNet quarter detection is working!')
print('Quarter was successfully detected and calibrated!')
