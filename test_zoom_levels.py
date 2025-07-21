#!/usr/bin/env python3
"""Quick visualization to understand zoom levels"""

import numpy as np
import matplotlib.pyplot as plt
from generate import MandelbrotLocationFinder

# Create finder
finder = MandelbrotLocationFinder(max_iterations=100)

# Test different zoom levels at the same location
x, y = -0.7269, 0.1889  # Classic location with nice structure
zoom_levels = [1, 10, 100, 1000, 10000]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, zoom in enumerate(zoom_levels):
    ax = axes[i]
    
    # Calculate bounds
    x_range = 3.5 / zoom
    y_range = 2.5 / zoom
    xmin = x - x_range/2
    xmax = x + x_range/2
    ymin = y - y_range/2
    ymax = y + y_range/2
    
    # Generate Mandelbrot
    mandelbrot = finder.calculate_mandelbrot(xmin, xmax, ymin, ymax, 256, 256)
    
    # Visualize with better colormap
    normalized = mandelbrot / finder.max_iterations
    sqrt_scaled = np.sqrt(normalized)
    
    ax.imshow(sqrt_scaled, cmap='twilight_shifted', extent=[xmin, xmax, ymin, ymax])
    ax.set_title(f'Zoom: {zoom}x\nRange: ±{x_range/2:.2e}', fontsize=12)
    ax.axis('off')

plt.suptitle(f'Zoom Level Comparison at ({x:.4f}, {y:.4f})', fontsize=16)
plt.tight_layout()
plt.savefig('zoom_level_comparison.png', dpi=150, bbox_inches='tight')
plt.show()