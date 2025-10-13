"""
Helper script to download SAM 2 model checkpoint.

Downloads SAM 2 models with support for different sizes:
- hiera_large: ~224M parameters, ~900MB (best quality)
- hiera_tiny: ~39M parameters, ~156MB (fastest, lower quality)
"""

import urllib.request
from pathlib import Path
import sys
import argparse

# Model configurations
MODELS = {
    "large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "name": "sam2_hiera_large.pt",
        "params": "224M",
        "size": "~900 MB",
        "description": "Best quality for fine hair frizz detection"
    },
    "tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "name": "sam2_hiera_tiny.pt",
        "params": "39M",
        "size": "~156 MB",
        "description": "Fastest model, lower quality"
    }
}

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
    parser = argparse.ArgumentParser(description="Download SAM 2 model checkpoint")
    parser.add_argument("--model", "-m", choices=["large", "tiny"], default="large",
                       help="Model size to download (default: large)")
    args = parser.parse_args()

    model_config = MODELS[args.model]

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / model_config["name"]

    # Check if already downloaded
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {model_path}")
        print(f"Size: {size_mb:.1f} MB")

        response = input(f"\nRe-download {args.model} model? (y/n): ").strip().lower()
        if response != 'y':
            print("Using existing model.")
            return

    print("=" * 70)
    print("SAM 2 Model Download")
    print("=" * 70)
    print(f"Model: hiera_{args.model} ({args.model.title()})")
    print(f"Parameters: {model_config['params']}")
    print(f"Size: {model_config['size']}")
    print(f"Description: {model_config['description']}")
    print("=" * 70)
    print()

    # Download
    success = download_with_progress(model_config["url"], model_path)

    if success:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\nModel saved to: {model_path}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"\nYou can now run the frizz analysis pipeline with {args.model} model")
    else:
        print("\nDownload failed. Please try:")
        print(f"1. Manually download from: {model_config['url']}")
        print(f"2. Save to: {model_path}")
        print("\nFor help with other models:")
        print("  python download_sam2_model.py --help")


if __name__ == "__main__":
    main()
