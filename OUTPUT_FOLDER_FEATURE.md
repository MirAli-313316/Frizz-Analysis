# Output Folder Selection & Timestamped Subfolders

## Overview
The application now supports custom output folder selection with automatic timestamped subfolder creation for better organization of analysis results.

## Features Implemented

### 1. Output Location Selection (GUI)
- **Location**: Top of left panel in GUI, above "Select Images"
- **Components**:
  - Read-only entry field displaying current output path
  - "Browse..." button to select output folder
  - Path validation with write-access testing
  - Persistent storage via configuration file

### 2. Timestamped Subfolders
- **Format**: `YYYY-MM-DD_HH-MM-SS` (e.g., `2025-10-03_14-30-25`)
- **Location**: Created inside selected base output folder
- **Contents**: All analysis outputs (Excel, visualizations, etc.)
- **Benefits**: 
  - No overwriting of previous results
  - Easy to track when analysis was run
  - Clean organization of multiple analysis sessions

### 3. Configuration Management
- **File**: `.app_config.json` (in project root)
- **Stores**:
  - `last_output_folder`: Last used output directory
  - `last_input_folder`: Last used input directory for file selection
- **Default**: `./outputs` if no config exists
- **Persistence**: Automatically saved when changed

### 4. User Experience Enhancements
- **Validation**: Output path is validated before processing starts
- **Status Display**: Full output path (including timestamp) shown during processing
- **Completion Dialog**: 
  - Shows full results path
  - Offers to open results folder
  - Auto-opens Excel file
- **Excel Report Tab**: 
  - Two buttons: "Open Excel File" and "Open Results Folder"
  - Easy access to both report and all output files
- **Error Handling**: 
  - Clear messages if output path is not writable
  - Graceful fallback to defaults if config is corrupted

## Usage

### Setting Output Location
1. Click "Browse..." button in Output Location section
2. Select desired base folder
3. Selection is saved for future sessions

### Processing Images
1. Select output location (optional - defaults to `./outputs`)
2. Select images to process
3. Click "Process Images"
4. Results are saved to: `[base_folder]/[timestamp]/`

### Example Output Structure
```
outputs/
├── 2025-10-03_14-30-25/
│   ├── test_results.xlsx
│   ├── analysis_IMG_8781.jpg
│   ├── analysis_IMG_8782.jpg
│   ├── calibration_IMG_8781.jpg
│   ├── calibration_IMG_8782.jpg
│   ├── segmentation_IMG_8781.jpg
│   └── segmentation_IMG_8782.jpg
└── 2025-10-03_15-45-10/
    ├── test_results.xlsx
    └── ...
```

## Technical Details

### New Files
- `src/config.py`: Configuration management module
  - `AppConfig` class for loading/saving preferences
  - JSON-based storage
  - Default values and error handling

### Modified Files
- `src/batch_processor.py`:
  - Added `create_timestamped_subfolder` parameter to `__init__`
  - Creates timestamped subfolder when requested
  - Uses `base_output_dir` + timestamp for organization

- `src/gui.py`:
  - Added output location selection UI
  - Integrated `AppConfig` for persistence
  - Updated processing to use timestamped folders
  - Enhanced completion dialog with folder opening
  - Fixed visualization/Excel tabs to use correct paths

- `.gitignore`:
  - Added `.app_config.json` to exclude user configs from git

## Configuration File Format
```json
{
  "last_output_folder": "./outputs",
  "last_input_folder": "C:\\Users\\user\\Documents\\Images"
}
```

## Error Handling
- **Invalid Path**: Shows error dialog, prevents processing
- **Not Writable**: Validates write access before processing
- **Config Corrupt**: Logs error, uses defaults
- **Missing Folder**: Automatically creates base directory

## Benefits
1. **Organization**: Each analysis run has its own folder
2. **No Overwrites**: Previous results are never lost
3. **Traceability**: Timestamp shows when analysis was performed
4. **Flexibility**: Users can organize results by project/experiment
5. **Persistence**: Settings remembered across sessions
6. **User-Friendly**: Clear visual feedback and easy access to results

