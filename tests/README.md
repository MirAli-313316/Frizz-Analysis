# Test Suite for Frizz Analysis Application

This folder contains test scripts and diagnostic tools for the hair frizz analysis pipeline.

## Test Files

### Integration Tests
- **`test_birefnet_integration.py`** - Complete pipeline test suite
  - Tests model loading, tress detection, segmentation, and analysis
  - Run with: `python tests/test_birefnet_integration.py`

### Component Tests
- **`test_quarter_detection.py`** - Quarter detection with visual outputs
  - Tests hybrid OpenCV + BiRefNet quarter detection
  - Generates visualization images in `outputs/quarter_test/`

### Diagnostic Tools
- **`diagnose_quarter_hybrid.py`** - Detailed quarter detection diagnostic
  - Shows step-by-step OpenCV and BiRefNet detection
  - Generates side-by-side comparison visualizations
  - Run with: `python tests/diagnose_quarter_hybrid.py [image_path]`

- **`diagnose_detection.py`** - General detection diagnostic tool

## Running Tests

### From Project Root
```bash
# Full integration test
python tests/test_birefnet_integration.py

# Quarter detection test
python tests/test_quarter_detection.py

# Detailed quarter diagnostic
python tests/diagnose_quarter_hybrid.py test_images/IMG_8787.JPG
```

### Test Outputs

All test outputs are saved to the `outputs/` directory (which is gitignored):
- `outputs/test_birefnet/` - Integration test outputs
- `outputs/quarter_test/` - Quarter detection visualizations
- `outputs/quarter_diagnostic/` - Detailed diagnostic visualizations

## Test Images

Test images are located in `test_images/` directory (gitignored by default).

## Adding New Tests

When adding new tests:
1. Name files with `test_*.py` or `diagnose_*.py` prefix
2. Place them in this `tests/` folder
3. Document them in this README
4. Ensure tests save outputs to `outputs/` directory (gitignored)

## CI/CD

These tests can be integrated into CI/CD pipelines for automated testing.

Note: Test files are tracked in Git, but test outputs (in `outputs/`) are not.

