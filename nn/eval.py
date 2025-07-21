#!/usr/bin/env python3
"""Evaluation and analysis for Mandelbrot location predictor."""

import argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

import config
import utils
from model import MandelbrotCNN
from train import DataLoader


def analyze_errors_by_zoom(predictions, targets, n_bins=10):
    """
    Analyze prediction errors binned by zoom level.
    
    Args:
        predictions: Predicted [x, y, log10_zoom]
        targets: Ground truth [x, y, log10_zoom]
        n_bins: Number of zoom bins
        
    Returns:
        Dictionary with error analysis
    """
    # Extract zoom values
    pred_zoom = predictions[:, 2]
    target_zoom = targets[:, 2]
    
    # Calculate errors
    coord_errors = np.sqrt(np.sum((predictions[:, :2] - targets[:, :2])**2, axis=1))
    zoom_errors = np.abs(pred_zoom - target_zoom)
    
    # Bin by target zoom
    zoom_min, zoom_max = target_zoom.min(), target_zoom.max()
    bins = np.linspace(zoom_min, zoom_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate error statistics per bin
    coord_error_by_bin = []
    zoom_error_by_bin = []
    counts_by_bin = []
    
    for i in range(n_bins):
        mask = (target_zoom >= bins[i]) & (target_zoom < bins[i+1])
        if mask.sum() > 0:
            coord_error_by_bin.append(coord_errors[mask].mean())
            zoom_error_by_bin.append(zoom_errors[mask].mean())
            counts_by_bin.append(mask.sum())
        else:
            coord_error_by_bin.append(0)
            zoom_error_by_bin.append(0)
            counts_by_bin.append(0)
    
    # Check for systematic bias
    zoom_bias = (pred_zoom - target_zoom).mean()
    
    return {
        'bin_centers': bin_centers,
        'coord_error_by_bin': np.array(coord_error_by_bin),
        'zoom_error_by_bin': np.array(zoom_error_by_bin),
        'counts_by_bin': np.array(counts_by_bin),
        'overall_coord_mae': np.mean(coord_errors),
        'overall_zoom_mae': np.mean(zoom_errors),
        'zoom_bias': zoom_bias,
        'coord_rmse': np.sqrt(np.mean((predictions[:, :2] - targets[:, :2])**2))
    }


def visualize_predictions(model, dataloader, n_samples=16, save_path=None):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained model
        dataloader: Data loader
        n_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    # Get samples
    images_list = []
    targets_list = []
    predictions_list = []
    
    for images, targets in dataloader:
        predictions = model(images)
        
        images_list.append(images)
        targets_list.append(targets)
        predictions_list.append(predictions)
        
        if len(images_list) * images.shape[0] >= n_samples:
            break
    
    # Concatenate and convert to numpy
    all_images = mx.concatenate(images_list, axis=0)[:n_samples]
    all_targets = mx.concatenate(targets_list, axis=0)[:n_samples]
    all_predictions = mx.concatenate(predictions_list, axis=0)[:n_samples]
    
    # Convert to numpy for plotting
    images_np = np.array(all_images)
    targets_np = np.array(all_targets)
    predictions_np = np.array(all_predictions)
    
    # Create figure
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Display image
        ax.imshow(images_np[i], cmap='hot', aspect='auto')
        
        # Denormalize coordinates
        true_x, true_y, true_zoom = utils.denormalize_coordinates(*targets_np[i])
        pred_x, pred_y, pred_zoom = utils.denormalize_coordinates(*predictions_np[i])
        
        # Calculate errors
        coord_error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        zoom_error = np.abs(np.log10(pred_zoom) - np.log10(true_zoom))
        
        # Add text
        title = f"True: ({true_x:.4f}, {true_y:.4f}, {true_zoom:.0f})\n"
        title += f"Pred: ({pred_x:.4f}, {pred_y:.4f}, {pred_zoom:.0f})\n"
        title += f"Err: coord={coord_error:.4f}, log_zoom={zoom_error:.3f}"
        
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def plot_error_analysis(error_analysis, save_path=None):
    """Plot error analysis by zoom level."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Coordinate error by zoom
    mask = error_analysis['counts_by_bin'] > 0
    ax1.bar(error_analysis['bin_centers'][mask], 
            error_analysis['coord_error_by_bin'][mask],
            width=0.1, alpha=0.7)
    ax1.set_xlabel('Log10(Zoom)')
    ax1.set_ylabel('Mean Coordinate Error')
    ax1.set_title('Coordinate Prediction Error by Zoom Level')
    ax1.grid(True, alpha=0.3)
    
    # Zoom error by zoom
    ax2.bar(error_analysis['bin_centers'][mask],
            error_analysis['zoom_error_by_bin'][mask],
            width=0.1, alpha=0.7, color='orange')
    ax2.set_xlabel('Log10(Zoom)')
    ax2.set_ylabel('Mean Log10(Zoom) Error')
    ax2.set_title('Zoom Prediction Error by Zoom Level')
    ax2.grid(True, alpha=0.3)
    
    # Add overall statistics
    fig.suptitle(f"Overall MAE - Coord: {error_analysis['overall_coord_mae']:.4f}, "
                 f"Zoom: {error_analysis['overall_zoom_mae']:.3f}, "
                 f"Zoom Bias: {error_analysis['zoom_bias']:.3f}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error analysis to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mandelbrot coordinate predictor")
    parser.add_argument("--resolution", type=str, default="320x240",
                        help="Evaluation resolution (default: 320x240 for 4:3 aspect ratio)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--visualize", type=int, default=16,
                        help="Number of samples to visualize (0 to skip)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Setup paths
    resolution = utils.get_resolution(args.resolution)
    dataset_dir = Path("nn/datasets")  # Resolution-independent datasets
    
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    # Load dataset
    data_path = dataset_dir / config.COORD_FILE_TEMPLATE.format(split=args.split)
    if not data_path.exists():
        print(f"Error: {args.split} dataset not found at {data_path}")
        return
    
    print(f"Loading {args.split} dataset from {data_path}")
    dataloader = DataLoader(str(data_path), config.BATCH_SIZE, resolution, shuffle=False)
    print(f"Loaded {dataloader.n_samples} samples")
    
    # Load model
    print(f"Loading model from {args.model}")
    model = MandelbrotCNN()
    model.load_weights(args.model)
    
    # Evaluate
    print("\nEvaluating model...")
    all_predictions = []
    all_targets = []
    
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        predictions = model(images)
        all_predictions.append(np.array(predictions))
        all_targets.append(np.array(targets))
    
    # Concatenate results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Analyze errors
    print("\nAnalyzing errors by zoom level...")
    error_analysis = analyze_errors_by_zoom(all_predictions, all_targets)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Dataset: {args.split}")
    print(f"Samples: {len(all_predictions)}")
    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    print(f"\nOverall Metrics:")
    print(f"  Coordinate MAE: {error_analysis['overall_coord_mae']:.4f}")
    print(f"  Coordinate RMSE: {error_analysis['coord_rmse']:.4f}")
    print(f"  Log10(Zoom) MAE: {error_analysis['overall_zoom_mae']:.3f}")
    print(f"  Zoom Bias: {error_analysis['zoom_bias']:+.3f}")
    
    # Save metrics
    if save_dir:
        metrics_path = save_dir / f"{args.split}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'split': args.split,
                'n_samples': len(all_predictions),
                'resolution': list(resolution),
                'overall_coord_mae': float(error_analysis['overall_coord_mae']),
                'overall_coord_rmse': float(error_analysis['coord_rmse']),
                'overall_zoom_mae': float(error_analysis['overall_zoom_mae']),
                'zoom_bias': float(error_analysis['zoom_bias'])
            }, f, indent=2)
        print(f"\nSaved metrics to {metrics_path}")
    
    # Visualizations
    if args.visualize > 0:
        print(f"\nGenerating visualizations...")
        
        # Sample predictions
        vis_path = save_dir / f"{args.split}_predictions.png" if save_dir else None
        visualize_predictions(model, dataloader, args.visualize, vis_path)
        
        # Error analysis plot
        error_path = save_dir / f"{args.split}_error_analysis.png" if save_dir else None
        plot_error_analysis(error_analysis, error_path)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()