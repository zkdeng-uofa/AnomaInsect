#!/usr/bin/env python3
"""
SubSampleData.py - Random Subsampling Tool for Image Datasets

This script performs random subsampling of image datasets while maintaining
directory structure. Useful for creating smaller datasets for testing or
when working with limited computational resources.

Usage:
    python SubSampleData.py --input_dir <input_path> --output_dir <output_path> --ratio 0.5
    
Example:
    python SubSampleData.py --input_dir data/full_dataset --output_dir data/subset --ratio 0.3
"""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


# Supported image extensions (case-insensitive)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.JPG', '.JPEG', '.PNG'}


def get_image_files(directory: Path) -> List[Path]:
    """
    Recursively find all image files in a directory.
    
    Args:
        directory: Path to search for images
        
    Returns:
        List of Path objects for all image files found
    """
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_files.append(Path(root) / file)
    return image_files


def analyze_dataset(input_dir: Path) -> Dict[str, List[Path]]:
    """
    Analyze dataset structure and count images per subdirectory.
    
    Args:
        input_dir: Root directory of the dataset
        
    Returns:
        Dictionary mapping subdirectory names to lists of image paths
    """
    dataset_structure = defaultdict(list)
    
    # Get all subdirectories
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        # No subdirectories, treat as flat structure
        images = get_image_files(input_dir)
        dataset_structure['root'] = images
    else:
        # Organize by subdirectory
        for subdir in subdirs:
            images = get_image_files(subdir)
            if images:  # Only include if there are images
                dataset_structure[subdir.name] = images
    
    return dict(dataset_structure)


def print_statistics(dataset_structure: Dict[str, List[Path]], title: str = "Dataset Statistics"):
    """
    Print detailed statistics about the dataset.
    
    Args:
        dataset_structure: Dictionary mapping subdirs to image lists
        title: Title for the statistics output
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    total_images = 0
    
    if len(dataset_structure) == 1 and 'root' in dataset_structure:
        # Flat structure
        count = len(dataset_structure['root'])
        total_images = count
        print(f"  Total Images: {count:,}")
    else:
        # Hierarchical structure
        for subdir_name in sorted(dataset_structure.keys()):
            count = len(dataset_structure[subdir_name])
            total_images += count
            print(f"  {subdir_name:30s}: {count:6,} images")
        print(f"  {'-'*58}")
        print(f"  {'Total':30s}: {total_images:6,} images")
    
    print(f"{'='*60}\n")
    
    return total_images


def subsample_dataset(
    dataset_structure: Dict[str, List[Path]], 
    ratio: float,
    seed: int = 42
) -> Dict[str, List[Path]]:
    """
    Randomly subsample images from each subdirectory.
    
    Args:
        dataset_structure: Dictionary mapping subdirs to image lists
        ratio: Sampling ratio (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with subsampled images
    """
    random.seed(seed)
    subsampled = {}
    
    for subdir_name, images in dataset_structure.items():
        # Calculate number of images to sample
        n_samples = max(1, int(len(images) * ratio))  # At least 1 image
        
        # Random sampling without replacement
        sampled_images = random.sample(images, n_samples)
        subsampled[subdir_name] = sampled_images
    
    return subsampled


def copy_subsampled_dataset(
    subsampled_structure: Dict[str, List[Path]],
    input_dir: Path,
    output_dir: Path
) -> None:
    """
    Copy subsampled images to output directory maintaining structure.
    
    Args:
        subsampled_structure: Dictionary mapping subdirs to sampled image lists
        input_dir: Original input directory
        output_dir: Destination directory
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    
    for subdir_name, images in subsampled_structure.items():
        if subdir_name == 'root':
            # Flat structure - copy to output root
            dest_dir = output_dir
        else:
            # Hierarchical structure - create subdirectory
            dest_dir = output_dir / subdir_name
            dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy each image
        for img_path in images:
            # Get relative path from input dir
            try:
                rel_path = img_path.relative_to(input_dir)
                dest_path = output_dir / rel_path
            except ValueError:
                # If relative path fails, use filename only
                dest_path = dest_dir / img_path.name
            
            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(img_path, dest_path)
            copied_count += 1
            
            # Progress indicator
            if copied_count % 100 == 0:
                print(f"  Copied {copied_count} images...", end='\r')
    
    print(f"  ✓ Copied {copied_count} images successfully" + " " * 20)


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Raises:
        ValueError: If arguments are invalid
    """
    # Check ratio
    if not 0 < args.ratio <= 1.0:
        raise ValueError(f"Ratio must be between 0 and 1.0, got {args.ratio}")
    
    # Check input directory exists
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input_dir}")
    
    # Check if output directory already exists
    if args.output_dir.exists() and not args.overwrite:
        if list(args.output_dir.iterdir()):  # Directory is not empty
            raise FileExistsError(
                f"Output directory already exists and is not empty: {args.output_dir}\n"
                f"Use --overwrite to replace it."
            )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Randomly subsample an image dataset while maintaining directory structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Subsample 50% of images
  python SubSampleData.py --input_dir data/full --output_dir data/half --ratio 0.5
  
  # Subsample 10% with custom seed
  python SubSampleData.py --input_dir data/full --output_dir data/small --ratio 0.1 --seed 123
  
  # Overwrite existing output directory
  python SubSampleData.py --input_dir data/full --output_dir data/subset --ratio 0.3 --overwrite
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=Path,
        required=True,
        help='Input directory containing images'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Output directory for subsampled dataset'
    )
    
    parser.add_argument(
        '--ratio',
        type=float,
        required=True,
        help='Sampling ratio (0.0 to 1.0). E.g., 0.5 = 50%% of images'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output directory if it exists'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate arguments
        validate_args(args)
        
        print("\n🔍 Analyzing input dataset...")
        print(f"Input Directory: {args.input_dir.absolute()}")
        
        # Analyze dataset
        dataset_structure = analyze_dataset(args.input_dir)
        
        if not dataset_structure:
            print("❌ No images found in input directory!")
            sys.exit(1)
        
        # Print original statistics
        total_original = print_statistics(dataset_structure, "Original Dataset")
        
        # Perform subsampling
        print(f"📊 Subsampling at {args.ratio*100:.1f}% ratio (seed={args.seed})...")
        subsampled_structure = subsample_dataset(dataset_structure, args.ratio, args.seed)
        
        # Print subsampled statistics
        total_sampled = print_statistics(subsampled_structure, "Subsampled Dataset")
        
        # Summary
        print(f"Summary:")
        print(f"  Original images:   {total_original:,}")
        print(f"  Subsampled images: {total_sampled:,}")
        print(f"  Reduction:         {total_original - total_sampled:,} images ({(1-args.ratio)*100:.1f}%)")
        print()
        
        if args.dry_run:
            print("🔍 DRY RUN - No files will be copied")
            print(f"Would copy {total_sampled:,} images to: {args.output_dir.absolute()}")
        else:
            # Copy files
            print(f"📁 Copying subsampled dataset to: {args.output_dir.absolute()}")
            
            # Remove output dir if overwrite is enabled
            if args.output_dir.exists() and args.overwrite:
                print(f"  🗑️  Removing existing output directory...")
                shutil.rmtree(args.output_dir)
            
            copy_subsampled_dataset(subsampled_structure, args.input_dir, args.output_dir)
            
            print(f"\n✅ Success! Subsampled dataset created at:")
            print(f"   {args.output_dir.absolute()}")
    
    except (ValueError, FileNotFoundError, NotADirectoryError, FileExistsError) as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
