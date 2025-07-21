#!/usr/bin/env python3
"""Evaluate the trained model using the game's scoring system."""

import numpy as np
import mlx.core as mx
from pathlib import Path
import json
from tqdm import tqdm

import utils
from model import MandelbrotCNN

def calculate_game_score(predicted_x, predicted_y, predicted_zoom, 
                        true_x, true_y, true_zoom, max_score=5000):
    """
    Calculate score using the same logic as the web game.
    Matches the scoring system in index.html.
    """
    # Calculate normalized distance by zoom level
    zoom_factor = true_zoom
    normalized_distance = np.sqrt((predicted_x - true_x)**2 + (predicted_y - true_y)**2) * zoom_factor
    
    # Exponential decay scoring with decayFactor = 200
    decay_factor = 200
    score_ratio = np.exp(-normalized_distance / decay_factor)
    
    # Calculate final score
    final_score = max_score * score_ratio
    
    # Apply rounding rules from the game
    if final_score >= 1:
        final_score = round(final_score)
    else:
        final_score = round(final_score * 100) / 100
        if final_score <= 0:
            final_score = 0.01
    
    return final_score, normalized_distance

def evaluate_on_test_set(model_path, test_path, resolution=(256, 256), num_samples=100):
    """Evaluate model on test set and calculate game scores."""
    
    # Load model
    model = MandelbrotCNN()
    model.load_weights(model_path)
    print(f"Loaded model from {model_path}")
    
    # Load test data
    data = utils.load_coordinates(test_path)
    x_coords = data['x']
    y_coords = data['y']
    zoom_values = data['zoom']
    
    # Limit samples if requested
    if num_samples and num_samples < len(x_coords):
        indices = np.random.choice(len(x_coords), num_samples, replace=False)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]
        zoom_values = zoom_values[indices]
    
    print(f"Evaluating on {len(x_coords)} test samples at {resolution[0]}x{resolution[1]} resolution")
    
    scores = []
    distances = []
    coord_errors = []
    zoom_errors = []
    
    # Process in batches for efficiency
    batch_size = 16
    for i in tqdm(range(0, len(x_coords), batch_size)):
        batch_end = min(i + batch_size, len(x_coords))
        batch_x = x_coords[i:batch_end]
        batch_y = y_coords[i:batch_end]
        batch_zoom = zoom_values[i:batch_end]
        
        # Generate images
        coordinates = list(zip(batch_x, batch_y, batch_zoom))
        images = utils.batch_generate_mandelbrot(
            coordinates,
            resolution[0],
            resolution[1]
        )
        
        # Get predictions
        predictions = model(images)
        
        # Denormalize predictions
        for j, (pred, true_coord) in enumerate(zip(predictions, coordinates)):
            pred_x_norm, pred_y_norm, pred_log_zoom = pred.tolist()
            true_x, true_y, true_zoom = true_coord
            
            # Denormalize coordinates
            pred_x, pred_y, pred_zoom = utils.denormalize_coordinates(
                pred_x_norm, pred_y_norm, pred_log_zoom
            )
            
            # Calculate game score
            score, norm_dist = calculate_game_score(
                pred_x, pred_y, pred_zoom,
                true_x, true_y, true_zoom
            )
            
            scores.append(score)
            distances.append(norm_dist)
            
            # Calculate raw errors for analysis
            coord_error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
            zoom_error = abs(np.log10(pred_zoom) - np.log10(true_zoom))
            coord_errors.append(coord_error)
            zoom_errors.append(zoom_error)
    
    # Calculate statistics
    scores = np.array(scores)
    distances = np.array(distances)
    coord_errors = np.array(coord_errors)
    zoom_errors = np.array(zoom_errors)
    
    # Score distribution
    perfect_scores = np.sum(scores == 5000)
    high_scores = np.sum(scores >= 4000)
    medium_scores = np.sum((scores >= 1000) & (scores < 4000))
    low_scores = np.sum((scores >= 1) & (scores < 1000))
    fractional_scores = np.sum(scores < 1)
    
    print("\n=== Game Score Evaluation ===")
    print(f"Average Score: {np.mean(scores):.1f} / 5000")
    print(f"Median Score: {np.median(scores):.1f}")
    print(f"Min Score: {np.min(scores):.2f}")
    print(f"Max Score: {np.max(scores):.0f}")
    
    print("\n=== Score Distribution ===")
    print(f"Perfect (5000): {perfect_scores} ({100*perfect_scores/len(scores):.1f}%)")
    print(f"High (≥4000): {high_scores} ({100*high_scores/len(scores):.1f}%)")
    print(f"Medium (1000-3999): {medium_scores} ({100*medium_scores/len(scores):.1f}%)")
    print(f"Low (1-999): {low_scores} ({100*low_scores/len(scores):.1f}%)")
    print(f"Fractional (<1): {fractional_scores} ({100*fractional_scores/len(scores):.1f}%)")
    
    print("\n=== Distance Metrics ===")
    print(f"Average Normalized Distance: {np.mean(distances):.3f}")
    print(f"Median Normalized Distance: {np.median(distances):.3f}")
    
    print("\n=== Raw Prediction Errors ===")
    print(f"Coordinate MAE: {np.mean(coord_errors):.4f}")
    print(f"Log10 Zoom MAE: {np.mean(zoom_errors):.4f}")
    
    # Score percentiles
    percentiles = [10, 25, 50, 75, 90]
    print("\n=== Score Percentiles ===")
    for p in percentiles:
        score_p = np.percentile(scores, p)
        if score_p >= 1:
            print(f"{p}th percentile: {score_p:.0f}")
        else:
            print(f"{p}th percentile: {score_p:.2f}")
    
    return {
        'scores': scores,
        'distances': distances,
        'coord_errors': coord_errors,
        'zoom_errors': zoom_errors
    }

def main():
    """Run evaluation on the trained model."""
    # Paths
    model_path = Path("nn/models/256/best_model.npz")
    test_path = Path("nn/datasets/coords_test.npz")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Train a model first with: uv run python nn/train.py --resolution 256 --samples 1000 --epochs 10")
        return
    
    if not test_path.exists():
        print(f"Error: Test dataset not found at {test_path}")
        print("Generate data first with: uv run python nn/data.py")
        return
    
    # Run evaluation
    results = evaluate_on_test_set(
        str(model_path),
        str(test_path),
        resolution=(256, 256),
        num_samples=100  # Evaluate on 100 random test samples
    )
    
    # Save detailed results
    output_path = Path("nn/models/256/evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            'average_score': float(np.mean(results['scores'])),
            'median_score': float(np.median(results['scores'])),
            'score_percentiles': {
                str(p): float(np.percentile(results['scores'], p))
                for p in [10, 25, 50, 75, 90]
            },
            'perfect_scores': int(np.sum(results['scores'] == 5000)),
            'total_samples': len(results['scores'])
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main()