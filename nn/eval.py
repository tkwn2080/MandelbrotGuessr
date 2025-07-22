#!/usr/bin/env python3
"""Evaluation and analysis for Mandelbrot location predictor with strategy inspection."""

import argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import seaborn as sns

import config
import utils
from model import MandelbrotCNN, ScreenLoss
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
    
    # Detect model type
    is_screen_loss = predictions_np.shape[1] == 5
    
    if is_screen_loss:
        # Detailed visualization for ScreenLoss model
        n_cols = 3
        n_rows = min(n_samples, 4)  # Limit rows for readability
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(n_samples, n_rows)):
            # Column 1: Input image
            ax1 = axes[i, 0]
            ax1.imshow(images_np[i], cmap='hot', aspect='auto')
            ax1.set_title(f'Sample {i+1}: Input Image', fontsize=10)
            ax1.axis('off')
            
            # Extract predictions
            view_x = predictions_np[i, 0]
            view_y = predictions_np[i, 1]
            log_zoom = predictions_np[i, 2]
            click_x = predictions_np[i, 3]
            click_y = predictions_np[i, 4]
            zoom = 10 ** log_zoom
            
            # Column 2: View selection visualization
            ax2 = axes[i, 1]
            # Show minimap with view box
            ax2.set_xlim(-2.5, 1.0)
            ax2.set_ylim(-1.25, 1.25)
            ax2.set_aspect('equal')
            
            # Denormalize
            view_center_x = view_x * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
            view_center_y = view_y * config.Y_MAX
            true_x = targets_np[i, 0] * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
            true_y = targets_np[i, 1] * config.Y_MAX
            
            # Draw view box
            view_width = 3.5 / zoom
            view_height = 2.5 / zoom
            rect = plt.Rectangle((view_center_x - view_width/2, view_center_y - view_height/2),
                               view_width, view_height, fill=False, edgecolor='blue', linewidth=2)
            ax2.add_patch(rect)
            
            # Mark true location
            ax2.plot(true_x, true_y, 'r*', markersize=10, label='True')
            ax2.plot(view_center_x, view_center_y, 'bo', markersize=8, label='View Center')
            ax2.set_title(f'View Selection (Zoom: {zoom:.0f}x)', fontsize=10)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Column 3: Click within view
            ax3 = axes[i, 2]
            # Calculate final position
            final_x, final_y = calculate_final_coords(view_x, view_y, zoom, click_x, click_y)
            distance = np.sqrt((final_x - true_x)**2 + (final_y - true_y)**2)
            score = 10000 * np.exp(-distance / 0.3)
            
            # Show click position
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.plot(click_x, click_y, 'go', markersize=10, label='Click')
            ax3.plot(0.5, 0.5, 'k+', markersize=15, label='Center')
            ax3.set_title(f'Click Position\nDist: {distance:.4f}, Score: {score:.0f}', fontsize=10)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
            ax3.set_aspect('equal')
            
    else:
        # Original visualization for standard model
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


def inspect_prediction_samples(model, dataloader, n_samples=5):
    """
    Detailed inspection of individual predictions for ScreenLoss models.
    Shows the complete decision process step by step.
    """
    samples_shown = 0
    
    for images, targets in dataloader:
        predictions = model(images)
        
        # Convert to numpy
        images_np = np.array(images)
        targets_np = np.array(targets)
        predictions_np = np.array(predictions)
        
        batch_size = images_np.shape[0]
        
        for i in range(min(batch_size, n_samples - samples_shown)):
            print(f"\n{'='*60}")
            print(f"SAMPLE {samples_shown + 1}")
            print(f"{'='*60}")
            
            # Extract all components
            if predictions_np.shape[1] == 5:
                view_x = predictions_np[i, 0]
                view_y = predictions_np[i, 1]
                log_zoom = predictions_np[i, 2]
                click_x = predictions_np[i, 3]
                click_y = predictions_np[i, 4]
                zoom = 10 ** log_zoom
                
                # Denormalize
                view_center_x = view_x * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
                view_center_y = view_y * config.Y_MAX
                true_x = targets_np[i, 0] * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
                true_y = targets_np[i, 1] * config.Y_MAX
                true_zoom = 10 ** targets_np[i, 2]
                
                # Calculate final position
                final_x, final_y = calculate_final_coords(view_x, view_y, zoom, click_x, click_y)
                
                # Calculate metrics
                view_error = np.sqrt((view_center_x - true_x)**2 + (view_center_y - true_y)**2)
                final_distance = np.sqrt((final_x - true_x)**2 + (final_y - true_y)**2)
                score = 10000 * np.exp(-final_distance / 0.3)
                
                # Pixel analysis
                pixel_x = click_x * 800
                pixel_y = click_y * 600
                rounded_pixel_x = np.round(pixel_x)
                rounded_pixel_y = np.round(pixel_y)
                
                print(f"\nTarget Location:")
                print(f"  Coordinates: ({true_x:.6f}, {true_y:.6f})")
                print(f"  Zoom: {true_zoom:.0f}x")
                
                print(f"\nModel Strategy:")
                print(f"  1. View Center: ({view_center_x:.6f}, {view_center_y:.6f})")
                print(f"     - Error from target: {view_error:.6f}")
                print(f"  2. Zoom Level: {zoom:.0f}x (log10: {log_zoom:.3f})")
                print(f"     - View shows: {3.5/zoom:.6f} x {2.5/zoom:.6f} region")
                print(f"  3. Click Position: ({click_x:.4f}, {click_y:.4f}) normalized")
                print(f"     - Pixels: ({pixel_x:.2f}, {pixel_y:.2f}) → ({rounded_pixel_x:.0f}, {rounded_pixel_y:.0f})")
                print(f"     - Distance from center: {np.sqrt((click_x-0.5)**2 + (click_y-0.5)**2):.3f}")
                
                print(f"\nFinal Result:")
                print(f"  Final coordinates: ({final_x:.6f}, {final_y:.6f})")
                print(f"  Distance to target: {final_distance:.6f}")
                print(f"  Game Score: {score:.0f} / 10000")
                
                # Check for suspicious behavior
                if zoom > 50000:
                    print(f"  ⚠️  WARNING: Extremely high zoom!")
                if score > 9900:
                    print(f"  ⚠️  WARNING: Suspiciously high score!")
                
            else:
                # Standard model
                pred_x, pred_y, pred_zoom = utils.denormalize_coordinates(*predictions_np[i])
                true_x, true_y, true_zoom = utils.denormalize_coordinates(*targets_np[i])
                
                distance = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
                score = 10000 * np.exp(-distance / 0.3)
                
                print(f"\nTarget: ({true_x:.6f}, {true_y:.6f}, {true_zoom:.0f}x)")
                print(f"Prediction: ({pred_x:.6f}, {pred_y:.6f}, {pred_zoom:.0f}x)")
                print(f"Distance: {distance:.6f}")
                print(f"Score: {score:.0f}")
            
            samples_shown += 1
            if samples_shown >= n_samples:
                return


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


def analyze_model_strategy(predictions, targets, model_type='screen_loss'):
    """
    Analyze the model's prediction strategies.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        model_type: 'screen_loss' (5 outputs) or 'standard' (3 outputs)
        
    Returns:
        Dictionary with strategy analysis
    """
    if model_type == 'screen_loss':
        # Extract components for ScreenLoss model
        view_x = predictions[:, 0]
        view_y = predictions[:, 1] 
        log_zoom = predictions[:, 2]
        click_x = predictions[:, 3]
        click_y = predictions[:, 4]
        
        # Convert to actual zoom values
        zoom = 10 ** log_zoom
        
        # Denormalize view centers
        view_center_x = view_x * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
        view_center_y = view_y * config.Y_MAX
        
        # Denormalize targets
        true_x = targets[:, 0] * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
        true_y = targets[:, 1] * config.Y_MAX
        true_zoom = 10 ** targets[:, 2]
        
        # Calculate view centering accuracy
        view_errors = np.sqrt((view_center_x - true_x)**2 + (view_center_y - true_y)**2)
        
        # Analyze click patterns
        click_center_dist = np.sqrt((click_x - 0.5)**2 + (click_y - 0.5)**2)
        
        # Calculate actual game scores using ScreenLoss logic
        screen_loss = ScreenLoss()
        scores = []
        final_distances = []
        
        for i in range(len(predictions)):
            # Calculate final coordinates after pixel discretization
            final_x, final_y = calculate_final_coords(
                view_x[i], view_y[i], zoom[i], click_x[i], click_y[i]
            )
            
            # Distance to target
            distance = np.sqrt((final_x - true_x[i])**2 + (final_y - true_y[i])**2)
            final_distances.append(distance)
            
            # Game score
            score = 10000 * np.exp(-distance / 0.3)
            scores.append(score)
        
        scores = np.array(scores)
        final_distances = np.array(final_distances)
        
        return {
            'model_type': 'screen_loss',
            'zoom_mean': np.mean(zoom),
            'zoom_median': np.median(zoom),
            'zoom_std': np.std(zoom),
            'zoom_min': np.min(zoom),
            'zoom_max': np.max(zoom),
            'log_zoom_distribution': log_zoom,
            'view_error_mean': np.mean(view_errors),
            'view_error_median': np.median(view_errors),
            'click_center_bias': np.mean(click_center_dist),
            'click_positions': np.stack([click_x, click_y], axis=1),
            'game_score_mean': np.mean(scores),
            'game_score_median': np.median(scores),
            'game_score_std': np.std(scores),
            'final_distance_mean': np.mean(final_distances),
            'perfect_scores': np.sum(scores >= 9900),
            'high_scores': np.sum(scores >= 5000),
            'scores': scores
        }
    else:
        # Standard 3-output model
        pred_x_norm = predictions[:, 0]
        pred_y_norm = predictions[:, 1]
        pred_log_zoom = predictions[:, 2]
        
        # Denormalize
        pred_x = pred_x_norm * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
        pred_y = pred_y_norm * config.Y_MAX
        pred_zoom = 10 ** pred_log_zoom
        
        true_x = targets[:, 0] * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
        true_y = targets[:, 1] * config.Y_MAX
        
        distances = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        scores = 10000 * np.exp(-distances / 0.3)
        
        return {
            'model_type': 'standard',
            'zoom_mean': np.mean(pred_zoom),
            'zoom_median': np.median(pred_zoom),
            'zoom_std': np.std(pred_zoom),
            'zoom_min': np.min(pred_zoom),
            'zoom_max': np.max(pred_zoom),
            'distance_mean': np.mean(distances),
            'game_score_mean': np.mean(scores),
            'scores': scores
        }


def calculate_final_coords(view_x, view_y, zoom, click_x, click_y):
    """Calculate final coordinates after pixel discretization."""
    # Denormalize view center
    view_center_x = view_x * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
    view_center_y = view_y * config.Y_MAX
    
    # View dimensions
    view_width = 3.5 / zoom
    view_height = 2.5 / zoom
    
    # Convert click to pixels
    pixel_x = np.round(click_x * 800)
    pixel_y = np.round(click_y * 600)
    
    # Convert pixels to offset
    offset_x = (pixel_x - 400) * view_width / 800
    offset_y = (pixel_y - 300) * view_height / 600
    
    # Final coordinates
    final_x = view_center_x + offset_x
    final_y = view_center_y + offset_y
    
    return final_x, final_y


def check_for_gaming_behavior(strategy_analysis):
    """
    Check for signs the model is exploiting the system.
    
    Returns:
        Dictionary with gaming indicators
    """
    gaming_indicators = {}
    
    if strategy_analysis['model_type'] == 'screen_loss':
        # Check if zoom is suspiciously high
        gaming_indicators['extreme_zoom'] = strategy_analysis['zoom_median'] > 50000
        gaming_indicators['max_zoom_ratio'] = np.sum(strategy_analysis['log_zoom_distribution'] > 4.5) / len(strategy_analysis['log_zoom_distribution'])
        
        # Check if scores are suspiciously high
        gaming_indicators['suspicious_scores'] = strategy_analysis['game_score_mean'] > 8000
        gaming_indicators['perfect_score_ratio'] = strategy_analysis['perfect_scores'] / len(strategy_analysis['scores'])
        
        # Check click patterns
        click_positions = strategy_analysis['click_positions']
        # Check if clicks are unnaturally precise (e.g., always exact pixel centers)
        click_x_decimal = click_positions[:, 0] * 800 % 1
        click_y_decimal = click_positions[:, 1] * 600 % 1
        gaming_indicators['unnatural_precision'] = np.mean(np.abs(click_x_decimal - 0.5) < 0.1) > 0.8
        
    else:
        gaming_indicators['extreme_zoom'] = strategy_analysis['zoom_median'] > 50000
        gaming_indicators['suspicious_scores'] = strategy_analysis['game_score_mean'] > 8000
    
    # Summary
    gaming_indicators['likely_gaming'] = any([
        gaming_indicators.get('extreme_zoom', False),
        gaming_indicators.get('suspicious_scores', False),
        gaming_indicators.get('perfect_score_ratio', 0) > 0.5
    ])
    
    return gaming_indicators


def visualize_strategy(strategy_analysis, save_path=None):
    """Create comprehensive strategy visualization."""
    if strategy_analysis['model_type'] == 'screen_loss':
        fig = plt.figure(figsize=(16, 10))
        
        # Zoom distribution
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(strategy_analysis['log_zoom_distribution'], bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(np.log10(strategy_analysis['zoom_median']), color='red', linestyle='--', label=f'Median: {strategy_analysis["zoom_median"]:.0f}x')
        ax1.set_xlabel('Log10(Zoom)')
        ax1.set_ylabel('Count')
        ax1.set_title('Zoom Distribution')
        ax1.legend()
        
        # Click position heatmap
        ax2 = plt.subplot(2, 3, 2)
        clicks = strategy_analysis['click_positions']
        heatmap, xedges, yedges = np.histogram2d(clicks[:, 0], clicks[:, 1], bins=20)
        sns.heatmap(heatmap.T, cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Click Count'})
        ax2.set_xlabel('Click X (normalized)')
        ax2.set_ylabel('Click Y (normalized)')
        ax2.set_title('Click Position Heatmap')
        ax2.invert_yaxis()
        
        # Score distribution
        ax3 = plt.subplot(2, 3, 3)
        scores = strategy_analysis['scores']
        ax3.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(strategy_analysis['game_score_median'], color='red', linestyle='--', 
                   label=f'Median: {strategy_analysis["game_score_median"]:.0f}')
        ax3.set_xlabel('Game Score')
        ax3.set_ylabel('Count')
        ax3.set_title('Score Distribution')
        ax3.legend()
        
        # View error vs zoom
        ax4 = plt.subplot(2, 3, 4)
        ax4.set_xlabel('Log10(Zoom)')
        ax4.set_ylabel('View Centering Error')
        ax4.set_title('View Accuracy vs Zoom')
        
        # Score vs zoom
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(strategy_analysis['log_zoom_distribution'], scores, alpha=0.5, s=10)
        ax5.set_xlabel('Log10(Zoom)')
        ax5.set_ylabel('Game Score')
        ax5.set_title('Score vs Zoom Level')
        
        # Statistics summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        stats_text = f"""Strategy Statistics:
        
Zoom: {strategy_analysis['zoom_min']:.0f}x - {strategy_analysis['zoom_max']:.0f}x
Median Zoom: {strategy_analysis['zoom_median']:.0f}x
Mean Score: {strategy_analysis['game_score_mean']:.0f}
Perfect Scores: {strategy_analysis['perfect_scores']} ({100*strategy_analysis['perfect_scores']/len(scores):.1f}%)
High Scores (≥5000): {strategy_analysis['high_scores']} ({100*strategy_analysis['high_scores']/len(scores):.1f}%)

View Error: {strategy_analysis['view_error_mean']:.4f}
Click Center Bias: {strategy_analysis['click_center_bias']:.3f}
Final Distance: {strategy_analysis['final_distance_mean']:.4f}"""
        
        ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', 
                transform=ax6.transAxes, family='monospace')
        
        plt.tight_layout()
    else:
        # Simpler visualization for standard model
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1.hist(np.log10(10**strategy_analysis['zoom_mean']), bins=30)
        ax1.set_xlabel('Log10(Zoom)')
        ax1.set_title('Zoom Distribution')
        
        ax2.hist(strategy_analysis['scores'], bins=30)
        ax2.set_xlabel('Game Score')
        ax2.set_title('Score Distribution')
        
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved strategy visualization to {save_path}")
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
    
    # Detect model type based on output shape
    model_type = 'screen_loss' if all_predictions.shape[1] == 5 else 'standard'
    print(f"\nDetected model type: {model_type} ({all_predictions.shape[1]} outputs)")
    
    # Analyze errors
    print("\nAnalyzing errors by zoom level...")
    error_analysis = analyze_errors_by_zoom(all_predictions, all_targets)
    
    # Analyze model strategy
    print("\nAnalyzing model strategy...")
    strategy_analysis = analyze_model_strategy(all_predictions, all_targets, model_type)
    
    # Check for gaming behavior
    print("\nChecking for gaming behavior...")
    gaming_indicators = check_for_gaming_behavior(strategy_analysis)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Dataset: {args.split}")
    print(f"Samples: {len(all_predictions)}")
    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    print(f"Model Type: {model_type}")
    print(f"\nOverall Metrics:")
    print(f"  Coordinate MAE: {error_analysis['overall_coord_mae']:.4f}")
    print(f"  Coordinate RMSE: {error_analysis['coord_rmse']:.4f}")
    print(f"  Log10(Zoom) MAE: {error_analysis['overall_zoom_mae']:.3f}")
    print(f"  Zoom Bias: {error_analysis['zoom_bias']:+.3f}")
    
    # Strategy summary
    print(f"\nStrategy Summary:")
    print(f"  Zoom Range: {strategy_analysis['zoom_min']:.0f}x - {strategy_analysis['zoom_max']:.0f}x")
    print(f"  Median Zoom: {strategy_analysis['zoom_median']:.0f}x")
    if model_type == 'screen_loss':
        print(f"  Mean Game Score: {strategy_analysis['game_score_mean']:.0f} / 10000")
        print(f"  Perfect Scores: {strategy_analysis['perfect_scores']} ({100*strategy_analysis['perfect_scores']/len(all_predictions):.1f}%)")
        print(f"  View Error: {strategy_analysis['view_error_mean']:.4f}")
        print(f"  Click Center Bias: {strategy_analysis['click_center_bias']:.3f}")
    else:
        print(f"  Mean Game Score: {strategy_analysis['game_score_mean']:.0f} / 10000")
    
    # Gaming behavior summary
    print(f"\nGaming Behavior Check:")
    if gaming_indicators['likely_gaming']:
        print("  ⚠️  WARNING: Model shows signs of gaming the system!")
        if gaming_indicators.get('extreme_zoom', False):
            print("     - Using extremely high zoom levels")
        if gaming_indicators.get('suspicious_scores', False):
            print("     - Achieving suspiciously high scores")
        if gaming_indicators.get('unnatural_precision', False):
            print("     - Clicking with unnatural precision")
    else:
        print("  ✓ No obvious gaming behavior detected")
    
    # Save metrics
    if save_dir:
        metrics_path = save_dir / f"{args.split}_metrics.json"
        with open(metrics_path, 'w') as f:
            metrics_dict = {
                'split': args.split,
                'n_samples': len(all_predictions),
                'resolution': list(resolution),
                'model_type': model_type,
                'overall_coord_mae': float(error_analysis['overall_coord_mae']),
                'overall_coord_rmse': float(error_analysis['coord_rmse']),
                'overall_zoom_mae': float(error_analysis['overall_zoom_mae']),
                'zoom_bias': float(error_analysis['zoom_bias']),
                'strategy': {
                    'zoom_median': float(strategy_analysis['zoom_median']),
                    'zoom_range': [float(strategy_analysis['zoom_min']), float(strategy_analysis['zoom_max'])],
                    'game_score_mean': float(strategy_analysis['game_score_mean'])
                },
                'gaming_behavior': gaming_indicators
            }
            if model_type == 'screen_loss':
                metrics_dict['strategy']['perfect_scores'] = int(strategy_analysis['perfect_scores'])
                metrics_dict['strategy']['view_error_mean'] = float(strategy_analysis['view_error_mean'])
                metrics_dict['strategy']['click_center_bias'] = float(strategy_analysis['click_center_bias'])
            # Convert any boolean values to Python bool for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, (np.bool_, np.bool)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(v) for v in obj]
                return obj
            
            json.dump(convert_for_json(metrics_dict), f, indent=2)
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
        
        # Strategy visualization
        strategy_path = save_dir / f"{args.split}_strategy.png" if save_dir else None
        visualize_strategy(strategy_analysis, strategy_path)
        
        # Detailed inspection of samples
        if model_type == 'screen_loss':
            print("\nDetailed sample inspection:")
            inspect_prediction_samples(model, dataloader, n_samples=5)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()