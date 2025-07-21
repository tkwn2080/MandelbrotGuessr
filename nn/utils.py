"""Utilities for MLX-accelerated Mandelbrot generation and coordinate handling."""

import mlx.core as mx
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm
import config
from scipy import ndimage


def mandelbrot_mlx(
    x_center: float,
    y_center: float,
    zoom: float,
    width: int,
    height: int,
    max_iterations: int = config.MAX_ITERATIONS
) -> mx.array:
    """
    Generate Mandelbrot set using MLX for GPU acceleration.
    
    Args:
        x_center: Center x coordinate
        y_center: Center y coordinate  
        zoom: Zoom level (not log scale)
        width: Image width
        height: Image height
        max_iterations: Maximum iterations
        
    Returns:
        MLX array of normalized iteration counts [0, 1]
    """
    # Calculate bounds based on zoom and aspect ratio
    # Base range for a square view
    base_range = 3.0 / zoom
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Adjust ranges to maintain correct aspect ratio
    if aspect_ratio >= 1:
        # Wider than tall (e.g., 800x600, 4:3)
        x_range = base_range * aspect_ratio
        y_range = base_range
    else:
        # Taller than wide
        x_range = base_range
        y_range = base_range / aspect_ratio
    
    xmin = x_center - x_range / 2
    xmax = x_center + x_range / 2
    ymin = y_center - y_range / 2
    ymax = y_center + y_range / 2
    
    # Create coordinate grids
    x = mx.linspace(xmin, xmax, width)
    y = mx.linspace(ymin, ymax, height)
    X, Y = mx.meshgrid(x, y)
    C = X + 1j * Y
    
    # Initialize arrays
    Z = mx.zeros(C.shape, dtype=mx.complex64)
    M = mx.zeros(C.shape, dtype=mx.int32)
    
    # Mandelbrot iteration
    for i in range(max_iterations):
        mask = mx.abs(Z) <= config.BAILOUT_RADIUS
        Z = mx.where(mask, Z * Z + C, Z)
        M = mx.where(mask, i, M)
    
    # Normalize to [0, 1]
    return M.astype(mx.float32) / max_iterations


def normalize_coordinates(x: float, y: float, zoom: float) -> Tuple[float, float, float]:
    """
    Normalize coordinates for neural network input.
    
    Args:
        x: X coordinate
        y: Y coordinate
        zoom: Zoom level (not log scale)
        
    Returns:
        Normalized (x, y, log10_zoom)
    """
    # Normalize x to [-1, 1] based on typical Mandelbrot bounds
    x_norm = (x - (config.X_MIN + config.X_MAX) / 2) / ((config.X_MAX - config.X_MIN) / 2)
    
    # Normalize y to [-1, 1] 
    y_norm = y / config.Y_MAX
    
    # Convert zoom to log scale
    log_zoom = np.log10(zoom)
    
    return x_norm, y_norm, log_zoom


def denormalize_coordinates(x_norm: float, y_norm: float, log_zoom: float) -> Tuple[float, float, float]:
    """
    Denormalize neural network output to actual coordinates.
    
    Args:
        x_norm: Normalized x
        y_norm: Normalized y  
        log_zoom: Log10 of zoom
        
    Returns:
        Actual (x, y, zoom)
    """
    # Denormalize x
    x = x_norm * ((config.X_MAX - config.X_MIN) / 2) + (config.X_MIN + config.X_MAX) / 2
    
    # Denormalize y
    y = y_norm * config.Y_MAX
    
    # Convert log zoom back to zoom
    zoom = 10 ** log_zoom
    
    return x, y, zoom


def sample_location(
    zoom_range: Tuple[float, float],
    respect_symmetry: bool = True
) -> Tuple[float, float, float]:
    """
    Sample a random location in the Mandelbrot set.
    
    Args:
        zoom_range: (min_log_zoom, max_log_zoom) 
        respect_symmetry: If True, only sample y > 0
        
    Returns:
        (x, y, zoom)
    """
    # Sample zoom uniformly in log space
    log_zoom = np.random.uniform(zoom_range[0], zoom_range[1])
    zoom = 10 ** log_zoom
    
    # Sample x uniformly
    x = np.random.uniform(config.X_MIN, config.X_MAX)
    
    # Sample y (respect symmetry if requested)
    if respect_symmetry:
        y = np.random.uniform(config.Y_MIN, config.Y_MAX)
    else:
        y = np.random.uniform(-config.Y_MAX, config.Y_MAX)
    
    return x, y, zoom


def generate_uniform_locations(
    num_samples: int,
    zoom_range: Tuple[float, float],
    max_iterations: int = 50,  # Fewer iterations for quick check
    validation_size: int = 16,  # Small size for validation
    min_entropy: float = 3.5,
    min_edge_density: float = 0.05,
    max_inside_ratio: float = 0.95
) -> list:
    """
    Generate uniformly distributed Mandelbrot locations with quality filtering.
    
    This ensures unbiased sampling across the entire complex plane,
    rejecting images that are too simple or boring based on entropy and edge density.
    
    Args:
        num_samples: Number of samples to generate
        zoom_range: (min_log_zoom, max_log_zoom)
        max_iterations: Iterations for validation check
        validation_size: Size of validation image
        min_entropy: Minimum Shannon entropy threshold
        min_edge_density: Minimum edge density threshold
        max_inside_ratio: Maximum ratio of pixels inside the set
        
    Returns:
        List of (x, y, zoom) tuples
    """
    locations = []
    attempts = 0
    rejected_uniform = 0
    rejected_quality = 0
    
    pbar = tqdm(total=num_samples, desc="Generating quality locations", unit="loc")
    
    while len(locations) < num_samples:
        # True uniform sampling
        x, y, zoom = sample_location(zoom_range, respect_symmetry=True)
        attempts += 1
        
        # Generate preview for validation (larger size for better quality metrics)
        preview_size = 64  # Bigger than before for better entropy/edge calculations
        preview = mandelbrot_mlx(x, y, zoom, preview_size, preview_size, max_iterations)
        preview_np = np.array(preview) * max_iterations  # Convert back to iteration counts
        
        # Check if not completely uniform
        unique_values = len(np.unique(preview_np))
        if unique_values <= 2:
            rejected_uniform += 1
            continue
        
        # Calculate quality metrics
        entropy = calculate_entropy(preview_np, max_iterations)
        edge_density = calculate_edge_density(preview_np)
        inside_ratio = np.sum(preview_np == max_iterations) / preview_np.size
        
        # Quality filtering
        if entropy < min_entropy or edge_density < min_edge_density or inside_ratio > max_inside_ratio:
            rejected_quality += 1
            continue
        
        locations.append((x, y, zoom))
        pbar.update(1)
        
        # Update statistics
        if len(locations) % 100 == 0:
            acceptance_rate = len(locations) / attempts if attempts > 0 else 0
            pbar.set_postfix({
                'accept': f'{acceptance_rate:.1%}',
                'rej_unif': rejected_uniform,
                'rej_qual': rejected_quality,
                'avg_zoom': f'{np.mean([loc[2] for loc in locations]):.0f}'
            })
    
    pbar.close()
    
    print(f"\nGenerated {num_samples} locations:")
    print(f"  Total attempts: {attempts}")
    print(f"  Rejected (uniform): {rejected_uniform}")
    print(f"  Rejected (quality): {rejected_quality}")
    print(f"  Acceptance rate: {len(locations)/attempts:.1%}")
    
    return locations


def calculate_entropy(image: np.ndarray, max_iterations: int) -> float:
    """
    Calculate Shannon entropy of a Mandelbrot image.
    
    Args:
        image: Array of iteration counts
        max_iterations: Maximum iteration count
        
    Returns:
        Shannon entropy value
    """
    # Normalize to 0-255 range
    normalized = ((image / max_iterations) * 255).astype(np.uint8)
    
    # Calculate histogram
    hist, _ = np.histogram(normalized.flatten(), bins=256, range=(0, 255))
    hist = hist[hist > 0]  # Remove zero entries
    
    if len(hist) == 0:
        return 0.0
    
    # Calculate probabilities
    prob = hist / hist.sum()
    
    # Calculate entropy
    entropy = -np.sum(prob * np.log2(prob))
    
    return entropy


def calculate_edge_density(image: np.ndarray) -> float:
    """
    Calculate edge density using Sobel edge detection.
    
    Args:
        image: Array of iteration counts
        
    Returns:
        Edge density (fraction of pixels that are edges)
    """
    from scipy import ndimage
    
    # Apply Sobel edge detection
    edges = ndimage.sobel(image.astype(float))
    
    # Calculate percentage of pixels that are edges
    threshold = np.percentile(np.abs(edges), 80)
    edge_pixels = np.sum(np.abs(edges) > threshold)
    total_pixels = edges.size
    
    return edge_pixels / total_pixels


def load_coordinates(filepath: str) -> dict:
    """Load coordinates from npz file."""
    data = np.load(filepath)
    return {
        'x': data['x'],
        'y': data['y'], 
        'zoom': data['zoom'],
        'log_zoom': data.get('log_zoom', np.log10(data['zoom']))
    }


def save_coordinates(
    filepath: str,
    x: np.ndarray,
    y: np.ndarray,
    zoom: np.ndarray
):
    """Save coordinates to npz file."""
    np.savez_compressed(
        filepath,
        x=x,
        y=y,
        zoom=zoom,
        log_zoom=np.log10(zoom)
    )


def batch_generate_mandelbrot(
    coordinates: list,
    width: int,
    height: int,
    max_iterations: int = config.MAX_ITERATIONS,
    show_progress: bool = False
) -> mx.array:
    """
    Generate batch of Mandelbrot images from coordinates.
    
    Args:
        coordinates: List of (x, y, zoom) tuples
        width: Image width
        height: Image height
        max_iterations: Maximum iterations
        show_progress: Whether to show progress bar
        
    Returns:
        Batch of images as MLX array [B, H, W]
    """
    images = []
    
    # Use progress bar if requested
    if show_progress:
        coordinates = tqdm(coordinates, desc="Generating Mandelbrot images")
    
    for x, y, zoom in coordinates:
        img = mandelbrot_mlx(x, y, zoom, width, height, max_iterations)
        images.append(img)
    
    return mx.stack(images)


def get_resolution(resolution_str: str) -> Tuple[int, int]:
    """Convert resolution string to (width, height) tuple."""
    if resolution_str in config.SUPPORTED_RESOLUTIONS:
        return config.SUPPORTED_RESOLUTIONS[resolution_str]
    
    # Try to parse custom resolution like "800x600"
    if 'x' in resolution_str:
        width, height = map(int, resolution_str.split('x'))
        return (width, height)
    
    # Single number means square resolution
    size = int(resolution_str)
    return (size, size)