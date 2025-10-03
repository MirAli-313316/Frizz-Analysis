"""Time point detection from filenames."""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from natsort import natsorted


@dataclass
class TimePoint:
    """Container for time point information."""
    
    hours: float
    label: str
    filename: str
    
    def __repr__(self):
        return f"TimePoint({self.label}, {self.hours}h)"
    
    @staticmethod
    def format_label(hours: float) -> str:
        """Format hours as readable label."""
        if hours == 0:
            return "0-hour"
        elif hours < 1:
            minutes = int(hours * 60)
            return f"{minutes}-min"
        elif hours == 1:
            return "1-hour"
        elif hours < 24:
            return f"{int(hours)}-hour"
        else:
            days = int(hours / 24)
            return f"{days}-day"

class TimePointParser:
    """Parse time points from image filenames."""
    
    # Standard time sequence for 8 images
    STANDARD_SEQUENCE = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 24.0]
    
    # Patterns for explicit time markers
    TIME_PATTERNS = {
        0.0: r'(0h|0-hour|0_hour|0hour|baseline|initial)',
        0.5: r'(30m|30-min|30_min|30min|0\.5h)',
        1.0: r'(1h|1-hour|1_hour|1hour)(?![\d])',  # Negative lookahead to avoid matching "1" in "10h"
        2.0: r'(2h|2-hour|2_hour|2hour)(?![\d])',
        4.0: r'(4h|4-hour|4_hour|4hour)',
        6.0: r'(6h|6-hour|6_hour|6hour)',
        8.0: r'(8h|8-hour|8_hour|8hour)',
        24.0: r'(24h|24-hour|24_hour|24hour|1d|1-day)',
    }
    
    @classmethod
    def parse_batch(cls, file_paths: List[str]) -> List[TimePoint]:
        """
        Parse time points for a batch of images.
        
        Args:
            file_paths: List of image file paths (strings)
        
        Returns:
            List of TimePoint objects
        """
        paths = [Path(p) for p in file_paths]
        time_points = []
        
        # First try explicit parsing
        explicit_times = []
        for path in paths:
            time = cls._parse_explicit(path.name)
            explicit_times.append(time)
        
        # If no explicit times found, use position-based
        if not any(t is not None for t in explicit_times):
            sorted_paths = natsorted(paths, key=lambda x: x.name)
            for i, path in enumerate(sorted_paths):
                if i < len(cls.STANDARD_SEQUENCE):
                    hours = cls.STANDARD_SEQUENCE[i]
                    time_points.append(TimePoint(
                        hours=hours,
                        label=TimePoint.format_label(hours),
                        filename=path.name
                    ))
        else:
            # Use explicit times
            for path, hours in zip(paths, explicit_times):
                if hours is not None:
                    time_points.append(TimePoint(
                        hours=hours,
                        label=TimePoint.format_label(hours),
                        filename=path.name
                    ))
        
        return time_points
    
    @classmethod
    def _parse_explicit(cls, filename: str) -> Optional[float]:
        """Try to parse explicit time from filename."""
        filename_lower = filename.lower()
        
        for hours, pattern in cls.TIME_PATTERNS.items():
            if re.search(pattern, filename_lower):
                return hours
        
        return None