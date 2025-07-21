#!/usr/bin/env python3
"""Test script to verify aspect ratio handling."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add nn directory to path
sys.path.append(str(Path(__file__).parent))

import utils
import config

def test_aspect_ratios():
    """Test that different aspect ratios show consistent views."""
    
    # Test location
    x, y, zoom = -0.5, 0.0, 1.0
    
    # Test resolutions
    resolutions = [
        ("256x256", 256, 256),
        ("320x240", 320, 240),
        ("640x480", 640, 480),
        ("800x600", 800, 600),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (name, width, height) in enumerate(resolutions):
        # Generate Mandelbrot
        img = utils.mandelbrot_mlx(x, y, zoom, width, height, max_iterations=100)
        img_np = np.array(img)
        
        # Display
        ax = axes[i]
        im = ax.imshow(img_np, cmap='hot', aspect='auto')
        ax.set_title(f'{name} (aspect {width/height:.2f})')
        ax.axis('off')
        
        # Add grid to show proportions
        ax.axhline(y=height/2, color='white', alpha=0.3, linestyle='--')
        ax.axvline(x=width/2, color='white', alpha=0.3, linestyle='--')
    
    plt.suptitle(f'Aspect Ratio Test - Location: ({x}, {y}) zoom={zoom}')
    plt.tight_layout()
    plt.savefig('aspect_ratio_test.png', dpi=150)
    plt.show()
    
    print("\nAspect ratio test complete!")
    print("Check aspect_ratio_test.png to verify all resolutions show the same view")
    print("\nKey points to verify:")
    print("- The Mandelbrot set should not appear stretched or squashed")
    print("- The same features should be visible in all images")
    print("- Only the resolution should differ, not the content")

if __name__ == "__main__":
    test_aspect_ratios()