"""
Helper script to download SAM 2 model checkpoint.

Downloads the SAM 2 hiera_large model (~224M parameters, ~900MB) 
suitable for RTX 3050 Ti (4GB VRAM) with best quality for hair frizz detection.
"""

import urllib.request
from pathlib import Path
import sys

MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
MODEL_NAME = "sam2_hiera_large.pt"

def download_with_progress(url: str, destination: Path):
    """Download file with progress bar."""
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 50
        filled = int(bar_length * downloaded / total_size)
        bar = '=' * filled + '-' * (bar_length - filled)
        
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        
        sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()
    
    print(f"Downloading {url}")
    print(f"Saving to: {destination}")
    print()
    
    try:
        urllib.request.urlretrieve(url, destination, show_progress)
        print("\n\nDownload complete!")
        return True
    except Exception as e:
        print(f"\n\nDownload failed: {e}")
        return False


def main():
    """Main download function."""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / MODEL_NAME
    
    # Check if already downloaded
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {model_path}")
        print(f"Size: {size_mb:.1f} MB")
        
        response = input("\nRe-download? (y/n): ").strip().lower()
        if response != 'y':
            print("Using existing model.")
            return
    
    print("=" * 70)
    print("SAM 2 Model Download")
    print("=" * 70)
    print(f"Model: hiera_large (Large)")
    print(f"Parameters: 224M")
    print(f"Size: ~900 MB")
    print(f"Suitable for: RTX 3050 Ti (4GB VRAM)")
    print(f"Best quality for fine hair frizz detection")
    print("=" * 70)
    print()
    
    # Download
    success = download_with_progress(MODEL_URL, model_path)
    
    if success:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\nModel saved to: {model_path}")
        print(f"Size: {size_mb:.1f} MB")
        print("\nYou can now run the frizz analysis pipeline")
    else:
        print("\nDownload failed. Please try:")
        print(f"1. Manually download from: {MODEL_URL}")
        print(f"2. Save to: {model_path}")
        print("\nAlternative models:")
        print("  - sam2_hiera_base_plus.pt (~80M params, faster)")
        print("  - sam2_hiera_small.pt (~46M params, fastest)")


if __name__ == "__main__":
    main()
