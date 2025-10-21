# Documentation & Examples

This folder contains documentation assets and example images for the Frizz Analysis application.

## Examples

The `examples/` folder contains sample images used in the README:

- **`input.jpg`** - Example input image showing 6 hair tresses with US quarter for calibration
- **`output.jpg`** - Analysis output showing color-coded segmented tresses with measurements

These images demonstrate the application's capabilities and are tracked in git for display on GitHub.

## Why These Images Are Tracked

Unlike `test_images/` (gitignored for being large test data) and `outputs/` (gitignored for being generated), the files in `docs/examples/` are:
- Optimized for web display
- Essential for documentation
- Small enough to track in git (~2-3 MB total)
- Used in README.md to showcase the tool

## Source

Example images generated from test image `IMG_8787.JPG`:
- Canon 18MP (5184×3456 pixels)
- 6 hair tresses
- US quarter in top-left corner
- Processing time: 3.8 seconds with GPU
- Total surface area: 547.94 cm²

