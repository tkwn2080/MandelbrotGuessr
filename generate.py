import numpy as np
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy import ndimage
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
from functools import partial
import time
import sys
from pathlib import Path

# Add nn directory to import the uniform generation
sys.path.append(str(Path(__file__).parent / "nn"))
import utils
import config

class MandelbrotLocationFinder:
    def __init__(self, max_iterations=100, n_workers=None):
        self.max_iterations = max_iterations
        self.n_workers = n_workers or multiprocessing.cpu_count()

    def calculate_mandelbrot(self, xmin, xmax, ymin, ymax, width, height):
        """Calculate Mandelbrot set for given bounds"""
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        Z = np.zeros_like(C)
        M = np.zeros(C.shape, dtype=int)

        for i in range(self.max_iterations):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            M[mask] = i

        return M

    def calculate_entropy(self, image):
        """Calculate Shannon entropy of an image to measure complexity"""
        # Normalize to 0-255 range
        normalized = ((image / self.max_iterations) * 255).astype(np.uint8)

        # Calculate histogram
        hist, _ = np.histogram(normalized.flatten(), bins=256, range=(0, 255))
        hist = hist[hist > 0]  # Remove zero entries

        # Calculate probabilities
        prob = hist / hist.sum()

        # Calculate entropy
        entropy = -np.sum(prob * np.log2(prob))

        return entropy

    def calculate_edge_density(self, image):
        """Calculate edge density as another measure of interest"""
        # Apply Sobel edge detection
        edges = ndimage.sobel(image.astype(float))

        # Calculate percentage of pixels that are edges
        threshold = np.percentile(np.abs(edges), 80)
        edge_pixels = np.sum(np.abs(edges) > threshold)
        total_pixels = edges.size

        return edge_pixels / total_pixels

    def is_interesting(self, image, min_entropy=4.0, min_edge_density=0.1):
        """Determine if a location is interesting enough"""
        entropy = self.calculate_entropy(image)
        edge_density = self.calculate_edge_density(image)

        # Check if not all black or all colored (boring areas)
        unique_values = len(np.unique(image))
        if unique_values < 5:
            return False, 0

        # Calculate overall interest score
        interest_score = (entropy / 8.0) * 0.6 + edge_density * 0.4

        return entropy >= min_entropy and edge_density >= min_edge_density, interest_score


    def find_uniform_locations(self, num_locations=1000, zoom_range=(1, 3), 
                             min_entropy=3.5, min_edge_density=0.05):
        """Find uniformly distributed locations with quality filtering"""
        print(f"Generating {num_locations} uniformly distributed locations...")
        print(f"Zoom range: 10^{zoom_range[0]:.1f} to 10^{zoom_range[1]:.1f}")
        print(f"Quality thresholds - Min entropy: {min_entropy}, Min edge density: {min_edge_density}")
        
        # Use the enhanced generation from nn/utils with quality filtering
        location_tuples = utils.generate_uniform_locations(
            num_locations, 
            zoom_range,
            max_iterations=self.max_iterations,
            validation_size=64,  # Larger for better quality metrics
            min_entropy=min_entropy,
            min_edge_density=min_edge_density,
            max_inside_ratio=0.95
        )
        
        # Convert to the format expected by the rest of the code
        locations = []
        for x, y, zoom in location_tuples:
            # Calculate a preview to get metrics for the score
            preview_size = 128
            x_range = 3.5 / zoom
            y_range = 2.5 / zoom
            xmin = x - x_range/2
            xmax = x + x_range/2
            ymin = y - y_range/2
            ymax = y + y_range/2
            
            preview = self.calculate_mandelbrot(xmin, xmax, ymin, ymax, preview_size, preview_size)
            entropy = self.calculate_entropy(preview)
            edge_density = self.calculate_edge_density(preview)
            score = (entropy / 8.0) * 0.6 + edge_density * 0.4
            
            locations.append({
                "x": x,
                "y": y,
                "zoom": zoom,
                "score": score,
                "entropy": entropy,
                "edge_density": edge_density
            })
        
        # Sort by score (higher score = more interesting/complex)
        locations.sort(key=lambda x: -x["score"])
        
        return locations


    def save_locations(self, locations, filename="mandelbrot_locations.json", save_js=True):
        """Save locations to JSON file with statistics and optionally as JS"""
        # Calculate statistics
        zoom_values = [loc["zoom"] for loc in locations]
        entropy_values = [loc.get("entropy", 0) for loc in locations]
        
        stats = {
            "total_locations": len(locations),
            "zoom_range": {
                "min": min(zoom_values),
                "max": max(zoom_values),
                "mean": np.mean(zoom_values)
            },
            "entropy_range": {
                "min": min(entropy_values),
                "max": max(entropy_values),
                "mean": np.mean(entropy_values)
            }
        }

        data = {
            "locations": locations,
            "statistics": stats
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved {len(locations)} locations to {filename}")
        print(f"Zoom range: {stats['zoom_range']['min']:.0f}x - {stats['zoom_range']['max']:.0f}x (avg: {stats['zoom_range']['mean']:.0f}x)")
        print(f"Entropy range: {stats['entropy_range']['min']:.2f} - {stats['entropy_range']['max']:.2f} (avg: {stats['entropy_range']['mean']:.2f})")
        
        # Also save as JS if requested (integrating convert.py functionality)
        if save_js:
            self.save_locations_js(locations)
    
    def save_locations_js(self, locations, filename="locations.js"):
        """Save locations as JavaScript file for web game"""
        js_content = "// Auto-generated Mandelbrot locations\n"
        js_content += "const MANDELBROT_LOCATIONS = [\n"
        
        for i, loc in enumerate(locations):
            js_content += "  {\n"
            js_content += f"    x: {loc['x']},\n"
            js_content += f"    y: {loc['y']},\n"
            js_content += f"    zoom: {loc['zoom']}\n"
            js_content += "  }"
            if i < len(locations) - 1:
                js_content += ","
            js_content += "\n"
        
        js_content += "];\n"
        
        with open(filename, 'w') as f:
            f.write(js_content)
        
        print(f"Saved {len(locations)} locations to {filename} for web game")

    def visualize_locations_grid(self, locations, num_samples=9, resolution=(256, 256)):
        """Visualize multiple locations in a grid"""
        num_samples = min(num_samples, len(locations))
        cols = 3
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        axes = axes.flatten() if num_samples > 1 else [axes]

        # Sample locations evenly distributed by score
        if len(locations) <= num_samples:
            sampled_locations = locations
        else:
            # Sample evenly across the score range
            indices = np.linspace(0, len(locations)-1, num_samples, dtype=int)
            sampled_locations = [locations[i] for i in indices]

        print("\nGenerating visualizations...")
        for idx, (ax, location) in enumerate(tqdm(zip(axes[:num_samples], sampled_locations))):
            width, height = resolution
            x_range = 3.5 / location["zoom"]
            y_range = 2.5 / location["zoom"]

            xmin = location["x"] - x_range/2
            xmax = location["x"] + x_range/2
            ymin = location["y"] - y_range/2
            ymax = location["y"] + y_range/2

            mandelbrot = self.calculate_mandelbrot(xmin, xmax, ymin, ymax, width, height)

            # Create custom colormap for better visualization
            # Normalize values and apply log scale for gradients
            normalized = mandelbrot / self.max_iterations
            # Apply sqrt scale for better visualization (less aggressive than log)
            sqrt_scaled = np.sqrt(normalized)
            
            # Use a perceptually uniform colormap
            im = ax.imshow(sqrt_scaled, cmap='twilight_shifted', extent=[xmin, xmax, ymin, ymax])
            
            # Build title with available metrics
            title_parts = [f'Zoom: {location["zoom"]:.0f}x']
            if 'entropy' in location:
                title_parts.append(f'Entropy: {location["entropy"]:.2f}')
            if 'edge_density' in location:
                title_parts.append(f'Edges: {location["edge_density"]:.1%}')
            ax.set_title('\n'.join(title_parts), fontsize=10)
            ax.axis('off')

        # Hide empty subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('mandelbrot_locations_grid.png', dpi=150, bbox_inches='tight')
        plt.show()

def benchmark_search_speed():
    """Quick benchmark to test search speed"""
    finder = MandelbrotLocationFinder(max_iterations=100)

    print("Running search speed benchmark...")
    print(f"Using {finder.n_workers} CPU cores")

    # Test with different resolutions
    for resolution in [(128, 128), (256, 256)]:
        print(f"\nTesting with resolution {resolution}:")
        start = time.time()
        locations = finder.find_interesting_locations_parallel(
            num_locations=50,
            resolution=resolution
        )
        elapsed = time.time() - start

        print(f"  Found {len(locations)} locations in {elapsed:.1f}s")
        print(f"  Average: {elapsed/len(locations):.2f}s per location")
        print(f"  Rate: {len(locations)/elapsed:.1f} locations/second")

def main():
    # Create finder instance
    finder = MandelbrotLocationFinder(max_iterations=100)

    # Generate uniform locations for web game
    # Match nn/config.py: zoom range 3-4 (10^3 to 10^4 = 1,000x to 10,000x)
    # This creates interesting, complex views of the Mandelbrot set
    print("Generating unbiased Mandelbrot locations for web game...")
    print("\nZoom explanation:")
    print("  - Zoom 1 = full Mandelbrot set view")
    print("  - Zoom 10 = 10x magnification")
    print("  - Zoom 1,000 = 1000x magnification (seeing fine details)")
    print("  - Zoom 10,000 = 10000x magnification (deep zoom, intricate patterns)")
    
    locations = finder.find_uniform_locations(
        num_locations=1000, 
        zoom_range=config.ZOOM_RANGE_TRAIN,  # Use (3, 4) from config
        min_entropy=3.5,
        min_edge_density=0.05
    )

    # Save as both JSON and JavaScript (integrated convert.py functionality)
    finder.save_locations(locations, save_js=True)

    # Visualize sample locations in a grid
    print("\nVisualizing sample locations...")
    finder.visualize_locations_grid(locations, num_samples=9)

if __name__ == "__main__":
    main()
