# Desktop Application Build Guide

This guide explains how to build the Frizz Analysis Tool as a standalone desktop application.

## Prerequisites

- Python 3.8+
- All dependencies listed in `requirements.txt`

## Quick Build

Run the automated build script:

```bash
python build_app.py
```

## Manual Build Steps

If you prefer to build manually:

1. **Install in development mode:**
   ```bash
   pip install -e .
   ```

2. **Build with PyInstaller:**
   ```bash
   pyinstaller app.spec --noconfirm
   ```

## Build Outputs

After building, you'll find:

### Windows
- `dist/FrizzAnalysis/FrizzAnalysis.exe` - Main executable
- `dist/FrizzAnalysis/` - Supporting files

### macOS
- `dist/Frizz Analysis.app` - Application bundle

### Linux
- `dist/FrizzAnalysis/FrizzAnalysis` - Standalone executable

## Icon Requirements

The application uses `icons/app_icon.ico` for Windows. Make sure this file exists and contains the appropriate icon sizes (16x16, 32x32, 48x48, etc.).

## Troubleshooting

### Common Issues

1. **Missing icon**: Ensure `icons/app_icon.ico` exists
2. **Import errors**: Run `pip install -e .` first to install the package
3. **Large file size**: PyInstaller includes all dependencies, resulting in larger executables

### Build Fails

Check the console output for specific error messages. Common issues:

- Missing dependencies: Install with `pip install -r requirements.txt`
- Path issues: Ensure you're running from the project root directory
- Icon file not found: Check that `icons/app_icon.ico` exists

## Distribution

For distributing to end users:

1. **Windows**: Share the entire `dist/FrizzAnalysis/` folder or create an installer
2. **macOS**: Share the `.app` bundle (may need code signing)
3. **Linux**: Share the standalone executable

## Development vs Production

- **Development**: Use `python run_gui.py` or `python -m src.gui`
- **Production**: Use the built executable from `dist/`

The built application is completely standalone and doesn't require Python installation on the target machine.
