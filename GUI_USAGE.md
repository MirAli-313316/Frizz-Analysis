# GUI Usage Guide

## Quick Start

1. **Launch the application**:
   ```bash
   python run_gui.py
   ```

2. **Select your images**:
   - Click "Select Images" button
   - Choose all time-series images from your experiment
   - Files can be in any order (will be auto-sorted)

3. **Review time points**:
   - Left panel shows detected time points
   - Format: "0-hour - filename.jpg"
   - Verify times are correct

4. **Process**:
   - Click "Process Images" button
   - Watch progress bar (real-time updates)
   - Wait for completion

5. **View results**:
   - Analysis tab: Surface area measurements
   - Visualization tab: Annotated images
   - Excel Report tab: Summary and export

## Interface Layout

### Top Bar
- **Title**: "Hair Frizz Analysis Tool"

### Left Panel: File Management
- **Select Images** button: Opens file chooser
- **Clear Selection** button: Resets current selection
- **File list**: Shows selected files with detected time points
- **Status**: Displays count of selected images

### Right Panel: Results (Tabs)

#### 1. Analysis Tab
Shows detailed results for each image:
- Image name
- Number of tresses detected
- Total surface area (cm²)
- Processing time
- Individual tress measurements

#### 2. Visualization Tab
Displays processed images:
- Dropdown menu to select which image to view
- Shows tress detection overlays
- Colored masks for each tress
- Surface area labels
- Quarter calibration circle

#### 3. Excel Report Tab
Summary of generated report:
- Location of Excel file
- "Open Excel File" button
- Preview of summary data table

### Bottom Panel: Controls
- **Process Images** button: Starts analysis (disabled until images selected)
- **Progress bar**: Shows completion percentage
- **Status text**: Current operation or ready state

## Features

### Automatic Time Detection
The GUI automatically detects time points from filenames:

**Recognized patterns**:
- `0-hour.jpg`, `0h.jpg` → 0-hour
- `30-min.jpg`, `30m.jpg` → 30-min  
- `1-hour.jpg`, `1h.jpg` → 1-hour
- `2-hour.jpg`, `2h.jpg` → 2-hour
- `4h.jpg`, `6h.jpg`, `8h.jpg` → 4-hour, 6-hour, 8-hour
- `24-hour.jpg`, `24h.jpg`, `1-day.jpg` → 24-hour

**Default naming** (IMG_XXXX.jpg):
- Sorted numerically and mapped to: 0h, 30min, 1h, 2h, 4h, 6h, 8h, 24h

### Background Processing
- Processing runs in background thread
- GUI remains responsive during analysis
- Can view other tabs while processing
- Progress updates in real-time

### Error Handling
- Individual image failures don't stop batch
- Error popups for critical issues
- Status messages for all operations
- Detailed logging to console

### Excel Auto-Open
- Excel report opens automatically when complete
- Works on Windows, macOS, and Linux
- Fallback if auto-open not supported

## Workflow Example

### Typical Experiment Session

1. **Prepare images**: Take photos at 0h, 30min, 1h, 2h, 4h, 6h, 8h
2. **Name files**: `0-hour.jpg`, `30-min.jpg`, `1-hour.jpg`, etc.
3. **Launch GUI**: `python run_gui.py`
4. **Select all images**: Click "Select Images" → Choose all 8 files
5. **Verify time points**: Check left panel shows correct sequence
6. **Process**: Click "Process Images"
7. **Wait**: Progress bar shows completion (typically 2-5 min for 8 images)
8. **Review**: 
   - Check Analysis tab for measurements
   - View Visualization tab to verify tress detection
   - Excel opens automatically with full report
9. **Repeat**: Click "Clear Selection" for next experiment

## Output Files

All outputs saved to `outputs/` directory:

### Visualizations
- `analysis_[name].jpg` - Full annotated image
- `calibration_[name].jpg` - Quarter detection check
- `segmentation_[name].jpg` - Mask overlay

### Excel Report
- `test_results.xlsx` - Multi-sheet workbook
  - **Summary**: All measurements by time point
  - **Change**: Percentage changes from baseline
  - **Statistics**: Summary stats per time point
  - **Metadata**: Processing details

## Tips

### Best Practices
- ✅ Select all time points from one experiment at once
- ✅ Use explicit time markers in filenames
- ✅ Verify quarter is visible in all images
- ✅ Check visualizations to confirm tress detection
- ✅ Keep Excel files for records

### Common Mistakes
- ❌ Selecting images from different experiments
- ❌ Missing calibration quarter in images
- ❌ Non-white backgrounds (affects segmentation)
- ❌ Interrupting processing mid-batch

## Keyboard Shortcuts
None currently implemented, but all operations accessible via buttons.

## Status Messages

| Message | Meaning |
|---------|---------|
| "Ready. Select images to begin." | Initial state, waiting for files |
| "Selected N images" | Files loaded, ready to process |
| "Starting processing..." | Beginning analysis |
| "Processing image N/M..." | Current progress |
| "Generating Excel report..." | Final report creation |
| "✓ Processing complete!" | Success, results ready |
| "❌ Error: ..." | Processing failed, see error details |

## Troubleshooting

### GUI won't start
```bash
# Check if customtkinter is installed
pip install customtkinter

# Run with error output
python run_gui.py
```

### Processing hangs
- Check console for error messages
- Verify images are valid JPEGs
- Ensure quarter is visible in all images

### Visualizations not showing
- Check `outputs/` directory exists
- Verify images were processed successfully
- Try restarting GUI

### Excel won't open
- Check file exists: `outputs/test_results.xlsx`
- Open manually from file explorer
- Verify Excel/LibreOffice installed

## Advanced

### Custom Output Directory
Edit `src/gui.py` line ~235:
```python
processor = BatchProcessor(output_dir="my_custom_outputs")
```

### Adjust Processing Size
Edit `src/gui.py` line ~257:
```python
max_processing_dim=1024  # Reduce if out of memory
```

### Change Excel Filename
Edit `src/gui.py` line ~297:
```python
output_filename="my_experiment.xlsx"
```

## Support

For technical issues, check:
1. Console output for error messages
2. `context/known_issues.md` for common problems
3. `context/project_spec.md` for detailed specifications

---

**Version**: 1.0  
**Last Updated**: October 2025


