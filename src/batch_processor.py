"""
Batch processor for time-series analysis of hair tress frizz testing.

This module processes multiple images over time, tracks surface area changes,
and generates comprehensive Excel reports with percentage changes from baseline.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

from .analysis import analyze_image, ImageAnalysis
from .time_parser import TimePointParser, TimePoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Process multiple images as a time-series analysis.
    
    Handles time point detection, baseline tracking, and percentage change calculations.
    """
    
    def __init__(self, output_dir: str = "outputs", create_timestamped_subfolder: bool = False):
        """
        Initialize batch processor.
        
        Args:
            output_dir: Base directory for output files
            create_timestamped_subfolder: If True, create a timestamped subfolder
        """
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subfolder if requested
        if create_timestamped_subfolder:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_dir = self.base_output_dir / timestamp
            self.output_dir.mkdir(exist_ok=True)
            logger.info(f"Created timestamped output folder: {self.output_dir}")
        else:
            self.output_dir = self.base_output_dir
        
        self.time_parser = TimePointParser()
        
    def process_time_series(
        self,
        image_paths: List[str],
        visualize: bool = True,
        num_expected_tresses: int = 7
    ) -> Tuple[List[ImageAnalysis], pd.DataFrame]:
        """
        Process a time series of images.
        
        Args:
            image_paths: List of image file paths
            visualize: Whether to generate visualizations
            num_expected_tresses: Expected number of tresses (for validation)
        
        Returns:
            Tuple of (analysis_results, summary_dataframe)
        """
        if not image_paths:
            raise ValueError("No images provided")
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"BATCH PROCESSING TIME SERIES")
        logger.info(f"{'=' * 70}")
        logger.info(f"Images: {len(image_paths)}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"{'=' * 70}\n")
        
        # Parse time points from filenames
        logger.info("Parsing time points from filenames...")
        time_points = self.time_parser.parse_batch(image_paths)
        
        # Sort by time
        sorted_items = sorted(zip(time_points, image_paths), key=lambda x: x[0].hours)
        time_points, image_paths = zip(*sorted_items)
        time_points = list(time_points)
        image_paths = list(image_paths)
        
        # Display time point assignments
        logger.info("\nTime point assignments:")
        for tp, path in zip(time_points, image_paths):
            logger.info(f"  {tp.label:>10} - {Path(path).name}")
        logger.info("")
        
        # Process each image
        results = []
        for tp, image_path in zip(time_points, image_paths):
            logger.info(f"\nProcessing {tp.label}...")
            try:
                result = analyze_image(
                    image_path,
                    visualize=visualize,
                    output_dir=str(self.output_dir),
                    num_expected_tresses=num_expected_tresses
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                logger.error("Continuing with next image...")
                continue
        
        if not results:
            raise RuntimeError("No images were successfully processed")
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Successfully processed {len(results)}/{len(image_paths)} images")
        logger.info(f"{'=' * 70}\n")
        
        # Create summary dataframe
        summary_df = self._create_summary_dataframe(results, time_points[:len(results)])
        
        return results, summary_df
    
    def generate_excel_report(
        self,
        results: List[ImageAnalysis],
        time_points: List[TimePoint],
        output_filename: str = "results.xlsx"
    ) -> Path:
        """
        Generate comprehensive Excel report with multiple sheets.
        
        Sheets:
        1. Summary - All surface areas by time point and tress
        2. Change - Percentage changes from baseline (0-hour)
        3. Statistics - Summary statistics
        4. Metadata - Processing information
        
        Args:
            results: List of ImageAnalysis results
            time_points: Corresponding time points
            output_filename: Output Excel filename
        
        Returns:
            Path to generated Excel file
        """
        output_path = self.output_dir / output_filename
        
        logger.info(f"Generating Excel report: {output_path}")
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Sheet 1: Summary - Raw surface areas
            summary_df = self._create_summary_sheet(results, time_points)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            logger.info("✓ Created Summary sheet")
            
            # Sheet 2: Change - Percentage changes from baseline
            if len(results) > 1:
                change_df = self._create_change_sheet(results, time_points)
                change_df.to_excel(writer, sheet_name='Change', index=False)
                logger.info("✓ Created Change sheet")
            
            # Sheet 3: Statistics
            stats_df = self._create_statistics_sheet(results, time_points)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            logger.info("✓ Created Statistics sheet")
            
            # Sheet 4: Metadata
            metadata_df = self._create_metadata_sheet(results, time_points)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            logger.info("✓ Created Metadata sheet")
        
        logger.info(f"✓ Excel report saved: {output_path}")
        return output_path
    
    def _create_summary_dataframe(
        self,
        results: List[ImageAnalysis],
        time_points: List[TimePoint]
    ) -> pd.DataFrame:
        """Create summary dataframe for quick viewing."""
        data = []
        
        for result, tp in zip(results, time_points):
            for tress in result.tresses:
                data.append({
                    'time_point': tp.label,
                    'hours': tp.hours,
                    'tress_id': tress.tress_id,
                    'area_cm2': round(tress.area_cm2, 2),
                    'pixels': tress.pixel_count
                })
        
        return pd.DataFrame(data)
    
    def _create_summary_sheet(
        self,
        results: List[ImageAnalysis],
        time_points: List[TimePoint]
    ) -> pd.DataFrame:
        """
        Create summary sheet with all surface areas.
        
        Format:
        Time Point | Tress 1 (cm²) | Tress 2 (cm²) | ... | Total (cm²)
        """
        # Determine max number of tresses
        max_tresses = max(len(r.tresses) for r in results)
        
        data = []
        for result, tp in zip(results, time_points):
            row = {
                'Time Point': tp.label,
                'Hours': tp.hours,
                'Image': result.image_name
            }
            
            # Add each tress
            for i in range(1, max_tresses + 1):
                tress = result.get_tress_by_id(i)
                if tress:
                    row[f'Tress {i} (cm²)'] = round(tress.area_cm2, 2)
                else:
                    row[f'Tress {i} (cm²)'] = None
            
            # Add total
            row['Total (cm²)'] = round(result.get_total_area(), 2)
            row['Tress Count'] = len(result.tresses)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _create_change_sheet(
        self,
        results: List[ImageAnalysis],
        time_points: List[TimePoint]
    ) -> pd.DataFrame:
        """
        Create change sheet with percentage changes from baseline (0-hour).
        
        Format:
        Time Point | Tress 1 (% change) | Tress 2 (% change) | ... | Total (% change)
        """
        if not results:
            return pd.DataFrame()
        
        # Get baseline (first time point, assumed to be 0-hour)
        baseline = results[0]
        
        # Determine max number of tresses
        max_tresses = max(len(r.tresses) for r in results)
        
        data = []
        for result, tp in zip(results, time_points):
            row = {
                'Time Point': tp.label,
                'Hours': tp.hours,
                'Image': result.image_name
            }
            
            # Calculate percentage change for each tress
            for i in range(1, max_tresses + 1):
                baseline_tress = baseline.get_tress_by_id(i)
                current_tress = result.get_tress_by_id(i)
                
                if baseline_tress and current_tress:
                    pct_change = ((current_tress.area_cm2 - baseline_tress.area_cm2) 
                                  / baseline_tress.area_cm2 * 100)
                    row[f'Tress {i} (% Δ)'] = round(pct_change, 2)
                else:
                    row[f'Tress {i} (% Δ)'] = None
            
            # Calculate total percentage change
            baseline_total = baseline.get_total_area()
            current_total = result.get_total_area()
            if baseline_total > 0:
                total_pct_change = ((current_total - baseline_total) 
                                   / baseline_total * 100)
                row['Total (% Δ)'] = round(total_pct_change, 2)
            else:
                row['Total (% Δ)'] = None
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _create_statistics_sheet(
        self,
        results: List[ImageAnalysis],
        time_points: List[TimePoint]
    ) -> pd.DataFrame:
        """Create statistics summary sheet."""
        data = []
        
        for result, tp in zip(results, time_points):
            areas = [t.area_cm2 for t in result.tresses]
            
            row = {
                'Time Point': tp.label,
                'Hours': tp.hours,
                'Tress Count': len(result.tresses),
                'Total Area (cm²)': round(sum(areas), 2),
                'Mean Area (cm²)': round(sum(areas) / len(areas), 2),
                'Min Area (cm²)': round(min(areas), 2),
                'Max Area (cm²)': round(max(areas), 2),
                'Std Dev (cm²)': round(pd.Series(areas).std(), 2),
                'Processing Time (s)': round(result.processing_time, 2)
            }
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _create_metadata_sheet(
        self,
        results: List[ImageAnalysis],
        time_points: List[TimePoint]
    ) -> pd.DataFrame:
        """Create metadata sheet with processing information."""
        data = []
        
        for result, tp in zip(results, time_points):
            row = {
                'Image Name': result.image_name,
                'Time Point': tp.label,
                'Hours': tp.hours,
                'Image Path': result.image_path,
                'Tress Count': len(result.tresses),
                'Calibration Factor (cm²/px)': f"{result.calibration_factor:.8f}",
                'Quarter Center X': result.quarter_info['center'][0],
                'Quarter Center Y': result.quarter_info['center'][1],
                'Quarter Radius (px)': round(result.quarter_info['radius'], 1),
                'Device Used': result.device_used,
                'Processing Time (s)': round(result.processing_time, 2)
            }
            
            data.append(row)
        
        # Add processing summary at the end
        data.append({})  # Blank row
        data.append({
            'Image Name': 'PROCESSING SUMMARY',
            'Time Point': '',
            'Hours': '',
        })
        data.append({
            'Image Name': 'Total Images Processed',
            'Time Point': len(results),
            'Hours': '',
        })
        data.append({
            'Image Name': 'Report Generated',
            'Time Point': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Hours': '',
        })
        
        return pd.DataFrame(data)


def process_directory(
    directory: str,
    pattern: str = "*.JPG",
    output_dir: str = "outputs",
    visualize: bool = True,
    excel_filename: str = "results.xlsx",
    max_processing_dim: int = 1024
) -> Tuple[List[ImageAnalysis], Path]:
    """
    Process all images in a directory matching the pattern.
    
    Args:
        directory: Directory containing images
        pattern: Glob pattern for image files (e.g., "*.JPG", "IMG_*.jpg")
        output_dir: Directory for outputs
        visualize: Whether to generate visualizations
        excel_filename: Name for Excel report
        max_processing_dim: Max dimension for SAM processing
    
    Returns:
        Tuple of (analysis_results, excel_report_path)
    """
    # Find all matching images
    directory = Path(directory)
    image_paths = sorted(directory.glob(pattern))
    
    if not image_paths:
        raise ValueError(f"No images found matching {pattern} in {directory}")
    
    logger.info(f"Found {len(image_paths)} images in {directory}")
    
    # Create batch processor
    processor = BatchProcessor(output_dir=output_dir)
    
    # Process time series
    results, summary_df = processor.process_time_series(
        [str(p) for p in image_paths],
        visualize=visualize,
        max_processing_dim=max_processing_dim
    )
    
    # Parse time points again for Excel generation
    time_parser = TimePointParser()
    time_points = time_parser.parse_batch([str(p) for p in image_paths])
    
    # Sort by time
    sorted_items = sorted(
        zip(time_points, results),
        key=lambda x: x[0].hours
    )
    time_points, results = zip(*sorted_items)
    time_points = list(time_points)
    results = list(results)
    
    # Generate Excel report
    excel_path = processor.generate_excel_report(
        results,
        time_points,
        output_filename=excel_filename
    )
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Processed: {len(results)} images")
    logger.info(f"Excel Report: {excel_path}")
    logger.info(f"Visualizations: {output_dir}/")
    logger.info(f"{'=' * 70}\n")
    
    return results, excel_path

