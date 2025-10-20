# Hair Frizz Analysis Tool

A scientific image analysis application for quantitative frizz testing of hair tresses using computer vision and segmentation models.

## Overview

This tool analyzes time-series images of hair tresses in humidity-controlled conditions to quantitatively measure anti-frizz product efficacy by calculating surface area changes over time.

## Features

- **Automatic Calibration**: Uses US quarter (24.26mm diameter) for pixel-to-cm² conversion
- **AI Segmentation**: BiRefNet model for accurate hair tress detection
- **Time-Series Analysis**: Automatically detects time points from filenames
- **Surface Area Calculation**: Precise measurements in cm²
- **Excel Reports**: Multi-sheet reports with percentage changes from baseline
- **Modern GUI**: User-friendly interface with real-time progress and visualization
- **Batch Processing**: Handle 1-10+ tresses across multiple time points

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, RTX 3050 Ti or better)
- 8GB+ RAM

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### GUI Application (Recommended)

Launch the graphical interface:

```bash
python run_gui.py
```

**GUI Features:**
1. **Select Images**: Click "Select Images" to choose your hair tress photos
2. **Auto Time Detection**: Time points are automatically detected from filenames
3. **Process**: Click "Process Images" to start analysis
4. **View Results**: 
   - Analysis tab: Surface area measurements
   - Visualization tab: Annotated images with tress detection
   - Excel Report tab: Summary statistics and export
5. **Auto-Open Excel**: Report opens automatically when complete

### Command Line

Process a directory of images:

```python
from src.batch_processor import process_directory

results, excel_path = process_directory(
    directory="test_images",
    pattern="*.JPG",
    output_dir="outputs",
    visualize=True,
    excel_filename="results.xlsx"
)
```

Process individual images:

```python
from src.analysis import analyze_image

result = analyze_image(
    "path/to/image.jpg",
    visualize=True,
    output_dir="outputs"
)

print(f"Total area: {result.get_total_area():.2f} cm²")
```

## Image Requirements

### Setup
- **Background**: White backdrop
- **Calibration**: US quarter in top-left corner of each image
- **Camera**: Any JPEG, recommended 12MP+ (tested with Canon 18MP)
- **Tresses**: 1-10+ tresses per image

### Filename Conventions

**Option 1**: Explicit time markers (recommended)
- `0-hour.jpg`, `30-min.jpg`, `1-hour.jpg`, `2-hour.jpg`, etc.
- Supports: `0h`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `24h`, `1d`

**Option 2**: Default camera names (auto-sequencing)
- `IMG_8780.JPG`, `IMG_8781.JPG`, `IMG_8782.JPG`, etc.
- Automatically mapped to standard sequence: 0h, 30min, 1h, 2h, 4h, 6h, 8h, 24h

## Output Files

### Visualizations (outputs/)
- `analysis_[name].jpg` - Annotated image with tress detection and measurements
- `calibration_[name].jpg` - Quarter detection visualization
- `segmentation_[name].jpg` - Segmentation masks

### Excel Report
Multi-sheet workbook containing:
1. **Summary**: Raw surface areas for all tresses by time point
2. **Change**: Percentage changes from baseline (0-hour)
3. **Statistics**: Mean, min, max, std dev per time point
4. **Metadata**: Processing details (calibration, device, timing)

## Project Structure

```
Frzz Analysis/
├── src/
│   ├── gui.py                 # Modern Tkinter GUI
│   ├── calibration.py         # Quarter detection
│   ├── segmentation.py        # BiRefNet hair detection
│   ├── analysis.py            # Surface area calculation
│   ├── batch_processor.py     # Time-series processing
│   └── time_parser.py         # Filename time detection
├── context/                   # Design docs and specifications
├── test_images/               # Sample images
├── outputs/                   # Generated results
├── run_gui.py                 # GUI launcher
└── requirements.txt           # Dependencies
```

## Technical Details

### Calibration
- Detects quarter using Hough Circle Transform
- Calculates cm²/pixel ratio per image
- US quarter diameter: 24.26mm
- Quarter area: 4.621 cm²

### Segmentation
- Uses BiRefNet_lite model for salient object detection
- GPU acceleration with CUDA (falls back to CPU)
- Memory-optimized processing (max 1024px dimension)
- CLAHE preprocessing for enhanced contrast and edge detection
- Filters out frame edges and background noise

### Measurement
- Surface area = pixel_count × calibration_factor
- Baseline tracking from 0-hour images
- Percentage change calculation: ((current - baseline) / baseline) × 100

## Hardware

**Tested Configuration:**
- GPU: NVIDIA GeForce RTX 3050 Ti (4GB VRAM)
- Automatic CUDA detection with CPU fallback

## Troubleshooting

### "No quarter detected"
- Ensure quarter is visible in top-left corner
- Check that background is white/light colored
- Quarter should be unobstructed and clearly visible

### "Out of memory"
- Close other GPU-intensive applications
- Use CPU mode (automatic fallback)
- BiRefNet is more memory efficient than previous models

### "Wrong time points"
- Use explicit filename markers (0-hour.jpg, 1-hour.jpg)
- Verify files are sorted correctly
- Check filename patterns match expected format

## Development

Run tests:
```bash
python test_calibration.py
python test_segmentation.py
python test_full_pipeline.py
```

## License

This is a scientific research tool. Please cite appropriately if used in publications.

## Support

For issues or questions, refer to the `context/` directory for detailed specifications and design decisions.


