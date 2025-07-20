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

    def process_location(self, params):
        """Process a single location (for parallel execution)"""
        region, resolution = params
        width, height = resolution

        if region.get("random_mode", False):
            # For random exploration: bias towards boundary by sampling from known interesting areas
            # Use a coarse calculation to find boundary points
            x_center = random.uniform(region["xmin"], region["xmax"])
            y_center = random.uniform(region["ymin"], region["ymax"])

            # Quick check if near boundary (low res)
            quick_check = self.calculate_mandelbrot(
                x_center - 0.01, x_center + 0.01,
                y_center - 0.01, y_center + 0.01,
                32, 32
            )

            # If not near boundary (all same value), try again
            if len(np.unique(quick_check)) < 3:
                return None

            # Random deep zoom between 1000x and 1000000x
            zoom = 10 ** random.uniform(3, 6)
        else:
            # Random location within the specified region
            x_center = random.uniform(region["xmin"], region["xmax"])
            y_center = random.uniform(region["ymin"], region["ymax"])

            # Random zoom level variation (keep it deep - between 0.8x and 5x of base zoom)
            zoom = region["zoom"] * random.uniform(0.8, 5.0)

        # Calculate bounds
        x_range = 3.5 / zoom
        y_range = 2.5 / zoom
        xmin = x_center - x_range/2
        xmax = x_center + x_range/2
        ymin = y_center - y_range/2
        ymax = y_center + y_range/2

        # Calculate Mandelbrot
        mandelbrot = self.calculate_mandelbrot(xmin, xmax, ymin, ymax, width, height)

        # Check if interesting
        is_good, score = self.is_interesting(mandelbrot)

        if is_good:
            return {
                "x": x_center,
                "y": y_center,
                "zoom": zoom,
                "score": score,
                "difficulty": self._calculate_difficulty(zoom)
            }
        return None

    def find_interesting_locations_parallel(self, num_locations=1000,
                                          batch_size=100,
                                          resolution=(256, 256)):
        """Find interesting locations using parallel processing"""
        locations = []

        # Hybrid approach: known interesting regions + random boundary exploration
        search_regions = [
            # Known deep regions (40% of searches)
            {"xmin": -0.7533, "xmax": -0.7532, "ymin": 0.1138, "ymax": 0.1139, "zoom": 10000, "weight": 0.1},
            {"xmin": 0.27507, "xmax": 0.27508, "ymin": 0.00628, "ymax": 0.00629, "zoom": 50000, "weight": 0.1},
            {"xmin": -0.088, "xmax": -0.087, "ymin": 0.654, "ymax": 0.655, "zoom": 1000, "weight": 0.05},
            {"xmin": -1.25066, "xmax": -1.25065, "ymin": -0.02012, "ymax": -0.02011, "zoom": 100000, "weight": 0.05},
            {"xmin": -0.7269, "xmax": -0.7268, "ymin": 0.1889, "ymax": 0.1890, "zoom": 10000, "weight": 0.05},
            {"xmin": -1.674, "xmax": -1.673, "ymin": 0.0001, "ymax": 0.0002, "zoom": 10000, "weight": 0.05},

            # Random exploration along main boundary (60% of searches)
            {"xmin": -2.0, "xmax": 0.5, "ymin": -1.25, "ymax": 1.25, "zoom": 1000, "weight": 0.6, "random_mode": True},
        ]

        # Create weighted region selection
        weights = [r["weight"] for r in search_regions]

        print(f"Starting parallel search with {self.n_workers} workers...")
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            with tqdm(total=num_locations, desc="Finding locations") as pbar:
                while len(locations) < num_locations:
                    # Create batch of tasks
                    current_batch_size = min(batch_size, num_locations - len(locations))
                    tasks = []

                    for _ in range(current_batch_size * 3):  # Over-sample to account for rejections
                        region = np.random.choice(search_regions, p=weights)
                        tasks.append((region, resolution))

                    # Submit batch for processing
                    futures = {executor.submit(self.process_location, task): task
                             for task in tasks}

                    # Collect results
                    for future in as_completed(futures):
                        result = future.result()
                        if result and len(locations) < num_locations:
                            locations.append(result)
                            pbar.update(1)
                            pbar.set_postfix({"score": f"{result['score']:.3f}",
                                            "zoom": f"{result['zoom']:.1f}"})

        elapsed = time.time() - start_time
        print(f"\nFound {len(locations)} locations in {elapsed:.1f} seconds")
        print(f"Average time per location: {elapsed/len(locations):.3f} seconds")

        # Sort by difficulty
        locations.sort(key=lambda x: x["difficulty"])

        return locations

    def _calculate_difficulty(self, zoom):
        """Calculate difficulty based on zoom level"""
        if zoom < 2:
            return 1  # Easy - full view
        elif zoom < 10:
            return 2  # Medium - slightly zoomed
        elif zoom < 100:
            return 3  # Hard - significantly zoomed
        elif zoom < 1000:
            return 4  # Very Hard - deep zoom
        else:
            return 5  # Extreme - ultra deep zoom

    def save_locations(self, locations, filename="mandelbrot_locations.json"):
        """Save locations to JSON file with statistics"""
        # Calculate statistics
        difficulties = [loc["difficulty"] for loc in locations]
        scores = [loc["score"] for loc in locations]

        stats = {
            "total_locations": len(locations),
            "difficulty_distribution": {
                i: difficulties.count(i) for i in range(1, 6)
            },
            "average_score": np.mean(scores),
            "score_range": [min(scores), max(scores)]
        }

        data = {
            "locations": locations,
            "statistics": stats
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved {len(locations)} locations to {filename}")
        print(f"Difficulty distribution: {stats['difficulty_distribution']}")
        print(f"Average score: {stats['average_score']:.3f}")

    def visualize_locations_grid(self, locations, num_samples=9, resolution=(256, 256)):
        """Visualize multiple locations in a grid"""
        num_samples = min(num_samples, len(locations))
        cols = 3
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        axes = axes.flatten() if num_samples > 1 else [axes]

        # Sample locations from different difficulty levels
        sampled_locations = []
        for difficulty in range(1, 6):
            diff_locs = [loc for loc in locations if loc["difficulty"] == difficulty]
            if diff_locs:
                sampled_locations.extend(random.sample(diff_locs,
                                       min(2, len(diff_locs))))

        sampled_locations = sampled_locations[:num_samples]

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

            im = ax.imshow(mandelbrot, cmap='hot', extent=[xmin, xmax, ymin, ymax])
            ax.set_title(f'Diff: {location["difficulty"]}, Zoom: {location["zoom"]:.1f}x\n'
                        f'Score: {location["score"]:.3f}', fontsize=10)
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
    # Create finder instance with parallel processing
    finder = MandelbrotLocationFinder(max_iterations=100)

    # Find interesting locations
    print("Searching for interesting Mandelbrot locations...")
    locations = finder.find_interesting_locations_parallel(num_locations=1000)

    # Save to file
    finder.save_locations(locations)

    # Visualize sample locations in a grid
    print("\nVisualizing sample locations...")
    finder.visualize_locations_grid(locations, num_samples=9)

if __name__ == "__main__":
    main()
