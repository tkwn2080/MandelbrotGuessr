#!/usr/bin/env python3
"""Data generation and management for Mandelbrot location predictor."""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Add parent directory to path to import from generate.py
sys.path.append(str(Path(__file__).parent.parent))

import config
import utils


def generate_locations(
    num_samples: int,
    zoom_range: tuple
) -> list:
    """
    Generate uniformly distributed Mandelbrot locations.
    
    Args:
        num_samples: Number of samples to generate
        zoom_range: (min_log_zoom, max_log_zoom)
        
    Returns:
        List of (x, y, zoom) tuples
    """
    print(f"Generating {num_samples} locations in zoom range 10^{zoom_range[0]:.1f} to 10^{zoom_range[1]:.1f}")
    
    # Use the new uniform sampling function
    locations = utils.generate_uniform_locations(num_samples, zoom_range)
    
    return locations


def generate_all_splits(
    num_samples: int
):
    """
    Generate all dataset splits automatically (resolution-independent).
    
    Args:
        num_samples: Total number of samples to generate
    
    Creates:
        - 80% train, 20% val from the main set (zoom 10^3 to 10^4)
        - Additional 20% as test set (zoom 10^4 to 10^4.3)
    """
    # Setup paths - no resolution needed!
    dataset_dir = Path("nn/datasets")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate main dataset (train + val)
    print(f"Generating {num_samples} samples for train/val split...")
    main_locations = generate_locations(num_samples, config.ZOOM_RANGE_TRAIN)
    
    # Shuffle and split 80/20
    np.random.shuffle(main_locations)
    split_idx = int(0.8 * len(main_locations))
    train_locations = main_locations[:split_idx]
    val_locations = main_locations[split_idx:]
    
    # Generate additional test set (20% of original number)
    test_samples = int(0.2 * num_samples)
    print(f"\nGenerating {test_samples} samples for test set...")
    test_locations = generate_locations(test_samples, config.ZOOM_RANGE_TEST1)
    
    # Save all splits
    for split_name, locations in [
        ("train", train_locations),
        ("val", val_locations), 
        ("test", test_locations)
    ]:
        x = np.array([loc[0] for loc in locations])
        y = np.array([loc[1] for loc in locations])
        zoom = np.array([loc[2] for loc in locations])
        
        filepath = dataset_dir / config.COORD_FILE_TEMPLATE.format(split=split_name)
        utils.save_coordinates(str(filepath), x, y, zoom)
        
        print(f"\n{split_name.capitalize()} set:")
        print(f"  Samples: {len(locations)}")
        print(f"  Zoom range: 10^{np.log10(zoom.min()):.2f} to 10^{np.log10(zoom.max()):.2f}")
        print(f"  Saved to: {filepath}")
    
    print(f"\nTotal samples generated: {len(main_locations) + len(test_locations)}")


def main():
    parser = argparse.ArgumentParser(description="Generate Mandelbrot dataset")
    parser.add_argument("--samples", type=int, required=True,
                        help="Number of samples to generate (will create 80/20 train/val + 20%% test)")
    
    args = parser.parse_args()
    
    # Generate all splits automatically
    generate_all_splits(args.samples)


if __name__ == "__main__":
    main()