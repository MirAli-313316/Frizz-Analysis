"""
Helper script to download SAM model checkpoint.

Downloads the SAM vit_b model (~375MB) suitable for RTX 3050 Ti (4GB VRAM).
"""

import urllib.request
from pathlib import Path
import sys

MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_NAME = "sam_vit_b.pth"

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
    print("SAM Model Download")
    print("=" * 70)
    print(f"Model: vit_b (Base)")
    print(f"Size: ~375 MB")
    print(f"Suitable for: RTX 3050 Ti (4GB VRAM)")
    print("=" * 70)
    print()
    
    # Download
    success = download_with_progress(MODEL_URL, model_path)
    
    if success:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\nModel saved to: {model_path}")
        print(f"Size: {size_mb:.1f} MB")
        print("\nYou can now run test_segmentation.py")
    else:
        print("\nDownload failed. Please try:")
        print(f"1. Manually download from: {MODEL_URL}")
        print(f"2. Save to: {model_path}")


if __name__ == "__main__":
    main()

