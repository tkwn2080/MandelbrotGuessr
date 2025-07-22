#\!/usr/bin/env python3
import numpy as np
import mlx.core as mx
from pathlib import Path
import utils
from model import MandelbrotCNN

# Load model
model = MandelbrotCNN()
model.load_weights("nn/models/256/best_model.npz")

# Load a few test samples
data = utils.load_coordinates("nn/datasets/coords_test.npz")
coords = list(zip(data['x'][:5], data['y'][:5], data['zoom'][:5]))

# Generate images and get predictions
images = utils.batch_generate_mandelbrot(coords, 256, 256)
predictions = model(images)

print("Sample predictions:")
for i, (true_coord, pred) in enumerate(zip(coords, predictions)):
    true_x, true_y, true_zoom = true_coord
    pred_x_norm, pred_y_norm, pred_log_zoom = pred.tolist()
    
    # Denormalize
    pred_x, pred_y, pred_zoom = utils.denormalize_coordinates(
        pred_x_norm, pred_y_norm, pred_log_zoom
    )
    
    print(f"\nSample {i+1}:")
    print(f"  True:      x={true_x:.4f}, y={true_y:.4f}, zoom={true_zoom:,}")
    print(f"  Predicted: x={pred_x:.4f}, y={pred_y:.4f}, zoom={pred_zoom:,.0f}")
    print(f"  Log10 zoom: {pred_log_zoom:.2f}")
    print(f"  Distance: {np.sqrt((pred_x-true_x)**2 + (pred_y-true_y)**2):.4f}")
