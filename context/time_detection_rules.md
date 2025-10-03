# Time Point Detection Rules

## Filename Parsing Priority

### Level 1: Explicit Time Markers
Look for these patterns in filename (case-insensitive):
- "0h", "0-hour", "0_hour", "0hour", "baseline", "initial" → 0 hours
- "30m", "30-min", "30_min", "30min", "0.5h" → 0.5 hours  
- "1h", "1-hour", "1_hour", "1hour" → 1 hour
- "2h", "2-hour", "2_hour", "2hour" → 2 hours
- "4h", "4-hour", "4_hour", "4hour" → 4 hours
- "6h", "6-hour", "6_hour", "6hour" → 6 hours
- "8h", "8-hour", "8_hour", "8hour" → 8 hours
- "24h", "24-hour", "24_hour", "24hour", "1d", "1-day" → 24 hours

### Level 2: Default Camera Names (IMG_XXXX)
If no time markers found:
1. Extract number from filename
2. Sort all images by number (ascending)
3. Map to standard sequence:
   - Image 1 (lowest) → 0 hours
   - Image 2 → 0.5 hours
   - Image 3 → 1 hour
   - Image 4 → 2 hours
   - Image 5 → 4 hours
   - Image 6 → 6 hours
   - Image 7 → 8 hours
   - Image 8 → 24 hours

### Level 3: User Override
Allow manual time assignment in GUI if needed

## Implementation Example
```python
import re

def extract_time_point(filename, position_in_batch=None):
    """Extract time point from filename or position."""
    
    # Check for explicit time markers
    patterns = {
        0.0: r'(0h|0-hour|0_hour|0hour|baseline|initial)',
        0.5: r'(30m|30-min|30_min|30min|0\.5h)',
        1.0: r'(1h|1-hour|1_hour|1hour)',
        # ... etc
    }
    
    for hours, pattern in patterns.items():
        if re.search(pattern, filename, re.IGNORECASE):
            return hours
    
    # Fallback to position-based
    if position_in_batch is not None:
        standard_sequence = [0, 0.5, 1, 2, 4, 6, 8, 24]
        if position_in_batch < len(standard_sequence):
            return standard_sequence[position_in_batch]
    
    return None