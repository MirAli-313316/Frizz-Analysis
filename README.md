# Hair Frizz Analysis Tool

A scientific image analysis application for quantitative frizz testing of hair tresses using computer vision and AI segmentation.

## Overview

This tool analyzes time-series images of hair tresses in humidity-controlled conditions to quantitatively measure anti-frizz product efficacy by calculating surface area changes over time.

## Features

- **Hybrid Quarter Detection**: OpenCV + BiRefNet for accurate calibration
- **AI Segmentation**: BiRefNet model for precise hair tress detection
- **Time-Series Analysis**: Automatically detects time points from filenames
- **Surface Area Calculation**: Precise measurements in cm²
- **Excel Reports**: Multi-sheet reports with percentage changes from baseline
- **Modern GUI**: User-friendly interface with real-time progress and visualization
- **Batch Processing**: Handle 1-10+ tresses across multiple time points
- **GPU Accelerated**: CUDA support with CPU fallback

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

The BiRefNet model will download automatically from Hugging Face on first use.

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
- **Calibration**: US quarter (24.26mm diameter) in top-left corner of each image
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
- `analysis_[name].jpg` - Annotated image with segmentation and measurements
- `tress_detection_[name].jpg` - Tress region detection visualization

### Excel Report
Multi-sheet workbook containing:
1. **Summary**: Raw surface areas for all tresses by time point
2. **Change**: Percentage changes from baseline (0-hour)
3. **Statistics**: Mean, min, max, std dev per time point
4. **Metadata**: Processing details (calibration, device, timing)

## Project Structure

```
Frizz Analysis/
├── src/                      # Core application code
│   ├── gui.py                # Modern Tkinter GUI
│   ├── calibration.py        # Hybrid quarter detection
│   ├── segmentation.py       # BiRefNet segmentation
│   ├── tress_detector.py     # OpenCV tress detection
│   ├── analysis.py           # Surface area calculation
│   ├── batch_processor.py    # Time-series processing
│   └── time_parser.py        # Filename time detection
├── tests/                    # Test suite (see tests/README.md)
│   ├── test_birefnet_integration.py
│   ├── test_quarter_detection.py
│   └── diagnose_quarter_hybrid.py
├── context/                  # Design docs and specifications
├── test_images/              # Sample images (gitignored)
├── outputs/                  # Generated results (gitignored)
├── run_gui.py                # GUI launcher
└── requirements.txt          # Dependencies
```

## Technical Details

### Quarter Detection (Hybrid Approach)
**Two-step process for accurate calibration:**
1. **OpenCV Hough Circle Detection** - Finds quarter location quickly
2. **BiRefNet Segmentation** - Precisely segments quarter for accurate area measurement

**Key parameters:**
- US quarter diameter: 24.26mm
- Quarter area: 4.621 cm²
- Circularity validation: >0.6 (ensures round object)
- Uses actual segmented area (not estimated circle area)

### Tress Segmentation
**OpenCV pre-detection + BiRefNet segmentation:**
1. **OpenCV** detects individual tress regions via binary thresholding
2. **BiRefNet** segments each tress crop for precise boundaries
3. **Filtering** removes noise and reconnects fragmented regions

**BiRefNet configuration:**
- Model: BiRefNet_lite (Hugging Face: ZhengPeng7/BiRefNet_lite)
- Input size: 1024x1024 (automatic resizing)
- Threshold: 0.35 (captures fine frizzy edges)
- CLAHE preprocessing for enhanced contrast
- GPU acceleration with CUDA (falls back to CPU)

### Measurement
- Surface area = pixel_count × calibration_factor
- Each image calibrated individually (accounts for camera movement)
- Baseline tracking from 0-hour images
- Percentage change: ((current - baseline) / baseline) × 100

## Hardware

**Tested Configuration:**
- GPU: NVIDIA GeForce RTX 3050 Ti (4GB VRAM)
- Automatic CUDA detection with CPU fallback
- ~0.17 GB VRAM for model
- ~3-4 seconds per 6-tress image with GPU

## Troubleshooting

### "No quarter detected"
- Ensure quarter is visible in top-left corner
- Check that background is white/light colored
- Quarter should be unobstructed and clearly visible
- Run diagnostic: `python tests/diagnose_quarter_hybrid.py [image]`

### "Out of memory"
- Close other GPU-intensive applications
- Use CPU mode (automatic fallback)
- BiRefNet uses ~0.17 GB VRAM (very efficient)

### "Wrong time points"
- Use explicit filename markers (0-hour.jpg, 1-hour.jpg)
- Verify files are sorted correctly
- Check filename patterns match expected format

### "Missing dependencies"
- Install requirements: `pip install -r requirements.txt`
- BiRefNet downloads automatically from Hugging Face
- Ensure `transformers` library installed

## Development & Testing

### Run Tests
```bash
# Full integration test
python tests/test_birefnet_integration.py

# Quarter detection with visualizations
python tests/test_quarter_detection.py

# Detailed diagnostic with step-by-step analysis
python tests/diagnose_quarter_hybrid.py [image_path]
```

See `tests/README.md` for detailed test documentation.

### Add New Tests
Place test scripts in `tests/` folder with `test_*.py` or `diagnose_*.py` prefix.

## Model Information

### BiRefNet
- **Source**: Hugging Face (ZhengPeng7/BiRefNet_lite)
- **Purpose**: Salient object detection with fine edge detail
- **License**: Check Hugging Face model card
- **First Run**: Model downloads automatically (~100 MB)

## License

This is a scientific research tool. Please cite appropriately if used in publications.

## Support

For detailed technical information:
- `context/project_spec.md` - Complete specifications
- `context/design_decisions.md` - Architecture choices
- `context/known_issues.md` - Resolved issues and solutions
- `tests/README.md` - Test suite documentation
