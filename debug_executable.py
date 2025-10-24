#!/usr/bin/env python3
"""
COMPREHENSIVE DEPENDENCY ANALYZER for PyInstaller executable.
This script analyzes ALL dependencies BEFORE building to prevent missing package issues.
"""

import sys
import traceback
import ast
import os
import re
import subprocess
from pathlib import Path
from typing import Set, List, Dict, Tuple


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory recursively."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    return python_files


def extract_imports_from_file(file_path: Path) -> Set[str]:
    """Extract all import statements from a Python file."""
    imports = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the AST to find imports
        try:
            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
                        # Also add specific imported items for better coverage
                        for alias in node.names:
                            imports.add(f"{node.module}.{alias.name}")

        except SyntaxError:
            # Fallback to regex if AST parsing fails
            import_lines = re.findall(r'^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))', content, re.MULTILINE)
            for match in import_lines:
                module = match[0] or match[1]
                imports.add(module)

    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")

    return imports


def analyze_codebase_dependencies() -> Set[str]:
    """Analyze all Python files in src/ to find all dependencies."""
    print("Analyzing codebase dependencies...")
    src_dir = Path("src")
    all_imports = set()

    if not src_dir.exists():
        print(f"[WARNING] src directory not found: {src_dir}")
        return all_imports

    python_files = find_python_files(src_dir)

    for file_path in python_files:
        imports = extract_imports_from_file(file_path)
        all_imports.update(imports)
        if imports:
            print(f"  {file_path.name}: {len(imports)} imports")

    print(f"Found {len(all_imports)} unique imports across {len(python_files)} files")
    return all_imports


def get_birefnet_dependencies() -> Set[str]:
    """Get BiRefNet model specific dependencies."""
    print("\nAnalyzing BiRefNet model dependencies...")

    birefnet_deps = set()

    try:
        # Load the model to see what it imports
        from transformers import AutoModelForImageSegmentation

        # Try to load BiRefNet to see what dependencies it needs
        print("  Loading BiRefNet model to analyze dependencies...")
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_lite",
            trust_remote_code=True
        )

        # Check what modules the model code imports by examining the module
        import inspect
        model_file = inspect.getfile(model.__class__)

        if model_file.startswith('C:') and 'huggingface' in model_file.lower():
            # Extract imports from the BiRefNet model file
            with open(model_file, 'r') as f:
                content = f.read()

            # Find import statements in the model file
            import_lines = re.findall(r'^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))', content, re.MULTILINE)
            for match in import_lines:
                module = match[0] or match[1]
                birefnet_deps.add(module)

            print(f"  BiRefNet model file: {model_file}")
            print(f"  BiRefNet dependencies: {len(birefnet_deps)} modules")

    except Exception as e:
        print(f"  Could not load BiRefNet model: {e}")
        # Add known dependencies based on error messages we've seen
        birefnet_deps.update(['timm', 'timm.models.layers', 'kornia'])

    return birefnet_deps


def test_module_import(module: str) -> Tuple[bool, str]:
    """Test if a module can be imported."""
    try:
        __import__(module)
        return True, ""
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Other error: {str(e)}"


def check_all_dependencies(imports: Set[str]) -> Dict[str, str]:
    """Check if all modules can be imported."""
    print("\nTesting module imports...")
    results = {}

    for module in sorted(imports):
        success, error = test_module_import(module)
        if success:
            results[module] = "OK"
        else:
            results[module] = f"FAIL: {error}"

    return results


def generate_pyinstaller_hidden_imports(imports: Set[str], results: Dict[str, str]) -> List[str]:
    """Generate PyInstaller hidden imports list."""
    hidden_imports = []

    # Filter out false positives (classes/functions that aren't modules)
    false_positives = {
        # Standard library false positives (classes/functions, not modules)
        'dataclasses.asdict', 'dataclasses.dataclass',
        'datetime.datetime', 'datetime.timedelta',
        'typing.Any', 'typing.Dict', 'typing.List', 'typing.Optional', 'typing.Tuple',

        # Application classes (not modules)
        'src.analysis.ImageAnalysis', 'src.analysis.analyze_image',
        'src.batch_processor.BatchProcessor',
        'src.calibration.detect_quarter',
        'src.config.AppConfig',
        'src.segmentation.SegmentationResult', 'src.segmentation.TressMask',
        'src.segmentation.load_birefnet_model', 'src.segmentation.segment_all_tresses', 'src.segmentation.visualize_segmentation',
        'src.time_parser.TimePoint', 'src.time_parser.TimePointParser',
        'src.tress_detector.detect_tress_regions', 'src.tress_detector.visualize_tress_detection',

        # BiRefNet model false positives (classes/functions within transformers)
        '.BiRefNet_config',
        'transformers.AutoModelForImageSegmentation',

        # Other function/class false positives
        'natsort.natsorted',
        'pathlib.Path',
    }

    # Add all modules that were successfully imported (excluding false positives)
    for module in sorted(imports):
        if (results[module] == "OK" and
            module not in false_positives and
            not module.startswith('.') and  # Remove relative imports
            '.' not in module):  # Remove class/function references
            hidden_imports.append(module)

    # Add specific modules that we know are needed (especially for ML models)
    essential_modules = [
        # Core ML frameworks (add these even if not detected)
        'torch._C',
        'torchvision._C',
        'transformers.models.auto',
        'transformers.modeling_utils',
        'transformers.models.auto.auto_factory',
        'transformers.dynamic_module_utils',

        # BiRefNet specific modules
        'timm.models',
        'timm.layers',
        'timm.models.layers',
        'timm.models.registry',
        'kornia.geometry',
        'kornia.filters',
        'einops',
        'huggingface_hub',
        'safetensors',
        'pyyaml',

        # GUI and core modules
        'PIL.Image',
        'cv2',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'seaborn'
    ]

    for module in essential_modules:
        success, _ = test_module_import(module)
        if success and module not in hidden_imports:
            hidden_imports.append(module)

    return sorted(list(set(hidden_imports)))


def run_pyinstaller_analysis():
    """Run PyInstaller analysis to see what it detects."""
    print("\nPyInstaller analysis...")
    print("Note: Run 'pyinstaller app.spec --dry-run' manually for detailed analysis")
    print("The comprehensive hidden imports in app.spec should cover all dependencies")

    # Just verify that the key modules are available
    key_modules = ['transformers', 'torch', 'torchvision', 'timm', 'kornia']
    available = []
    missing = []

    for module in key_modules:
        success, _ = test_module_import(module)
        if success:
            available.append(module)
        else:
            missing.append(module)

    if available:
        print(f"[OK] Available modules: {', '.join(available)}")
    if missing:
        print(f"[FAIL] Missing modules: {', '.join(missing)}")
    else:
        print("[SUCCESS] All key modules are available!")


def main():
    """Run comprehensive dependency analysis."""
    print("=" * 70)
    print("COMPREHENSIVE DEPENDENCY ANALYSIS FOR PYINSTALLER")
    print("=" * 70)

    # Step 1: Analyze codebase imports
    all_imports = analyze_codebase_dependencies()

    # Step 2: Get BiRefNet specific dependencies
    birefnet_deps = get_birefnet_dependencies()
    all_imports.update(birefnet_deps)

    # Step 3: Test all imports
    import_results = check_all_dependencies(all_imports)

    # Step 4: Report results
    print("\n" + "=" * 70)
    print("IMPORT TEST RESULTS")
    print("=" * 70)

    failed_imports = []
    for module, result in import_results.items():
        status = "[OK]" if result == "OK" else "[FAIL]"
        print(f"{status} {module}")
        if result != "OK":
            print(f"    Error: {result}")
            failed_imports.append(module)

    # Step 5: Generate PyInstaller configuration
    print("\n" + "=" * 70)
    print("GENERATED PYINSTALLER HIDDEN IMPORTS")
    print("=" * 70)

    hidden_imports = generate_pyinstaller_hidden_imports(all_imports, import_results)

    print(f"Total hidden imports needed: {len(hidden_imports)}")
    print("\nAdd these to your app.spec hiddenimports:")
    print("hiddenimports=[")

    for i, module in enumerate(hidden_imports, 1):
        comma = "," if i < len(hidden_imports) else ""
        print(f"    '{module}'{comma}")

    # Add specific imports for BiRefNet
    print("    # BiRefNet specific imports")
    birefnet_specific = [
        'timm.models.layers',
        'timm.models.registry',
        'transformers.models.auto.auto_factory',
        'transformers.dynamic_module_utils',
        'kornia.geometry',
        'huggingface_hub'
    ]

    for module in birefnet_specific:
        comma = "," if module != birefnet_specific[-1] else ""
        print(f"    '{module}'{comma}")

    print("],")

    # Step 6: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if failed_imports:
        print(f"[WARNING] {len(failed_imports)} modules failed to import:")
        for module in failed_imports[:10]:  # Show first 10
            print(f"  - {module}")
        if len(failed_imports) > 10:
            print(f"  ... and {len(failed_imports) - 10} more")

        print("\nInstall missing packages with:")
        print("pip install " + " ".join(failed_imports))
    else:
        print("[SUCCESS] All dependencies are available!")

    print(f"\nTotal modules analyzed: {len(all_imports)}")
    print(f"Hidden imports for PyInstaller: {len(hidden_imports)}")

    # Step 7: Run PyInstaller analysis
    run_pyinstaller_analysis()

    return 0 if not failed_imports else 1


if __name__ == "__main__":
    sys.exit(main())
