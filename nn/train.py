#!/usr/bin/env python3
"""Training pipeline for Mandelbrot location predictor."""

import argparse
import os
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from tqdm import tqdm
import json

import config
import utils
from model import MandelbrotCNN, MandelbrotLoss


class DataLoader:
    """Simple data loader for coordinate datasets."""
    
    def __init__(self, filepath: str, batch_size: int, resolution: tuple, shuffle: bool = True):
        self.batch_size = batch_size
        self.resolution = resolution
        self.shuffle = shuffle
        
        # Load coordinates
        data = utils.load_coordinates(filepath)
        self.x = data['x']
        self.y = data['y']
        self.zoom = data['zoom']
        self.log_zoom = data['log_zoom']
        self.n_samples = len(self.x)
        
        # Create indices
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            
            # Get coordinates for batch
            batch_x = self.x[batch_indices]
            batch_y = self.y[batch_indices]
            batch_zoom = self.zoom[batch_indices]
            
            # Generate images on-the-fly
            coordinates = list(zip(batch_x, batch_y, batch_zoom))
            images = utils.batch_generate_mandelbrot(
                coordinates,
                self.resolution[0],
                self.resolution[1]
            )
            
            # Normalize coordinates for targets
            targets = []
            for x, y, zoom in coordinates:
                x_norm, y_norm, log_zoom = utils.normalize_coordinates(x, y, zoom)
                targets.append([x_norm, y_norm, log_zoom])
            
            targets = mx.array(targets, dtype=mx.float32)
            
            yield images, targets


def evaluate(model, dataloader, loss_fn):
    """Evaluate model on a dataset."""
    total_loss = 0
    total_coord_error = 0
    total_zoom_error = 0
    n_batches = 0
    
    for images, targets in dataloader:
        # Forward pass
        predictions = model(images)
        loss = loss_fn(predictions, targets)
        
        # Calculate errors
        coord_error = mx.mean(mx.abs(predictions[:, :2] - targets[:, :2]))
        zoom_error = mx.mean(mx.abs(predictions[:, 2] - targets[:, 2]))
        
        total_loss += loss.item()
        total_coord_error += coord_error.item()
        total_zoom_error += zoom_error.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'coord_mae': total_coord_error / n_batches,
        'zoom_mae': total_zoom_error / n_batches
    }


def train_epoch(model, optimizer, train_loader, loss_fn, epoch):
    """Train for one epoch with gradient clipping."""
    total_loss = 0
    n_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for i, (images, targets) in enumerate(pbar):
        # Define loss computation function for this batch
        def compute_loss(model, images, targets):
            predictions = model(images)
            return loss_fn(predictions, targets)
        
        # Compute loss and gradients
        loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
        loss_value, grads = loss_and_grad_fn(model, images, targets)
        
        # Gradient clipping
        if config.GRADIENT_CLIP_NORM > 0:
            # Calculate gradient norm using tree_flatten
            total_norm = 0
            for _, grad in tree_flatten(grads):
                if grad is not None:
                    total_norm += mx.sum(grad ** 2).item()
            total_norm = total_norm ** 0.5
            
            # Clip gradients if norm exceeds threshold
            if total_norm > config.GRADIENT_CLIP_NORM:
                clip_scale = config.GRADIENT_CLIP_NORM / total_norm
                # Apply clipping to all gradients in the tree
                from mlx.utils import tree_map
                grads = tree_map(lambda g: g * clip_scale if g is not None else None, grads)
        
        # Update parameters
        optimizer.update(model, grads)
        
        # Update metrics
        total_loss += loss_value.item()
        avg_loss = total_loss / (i + 1)
        
        # Update progress bar with more info
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{optimizer.learning_rate:.2e}'
        })
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Mandelbrot coordinate predictor")
    parser.add_argument("--resolution", type=str, default="320x240",
                        help="Training resolution (default: 320x240 for 4:3 aspect ratio)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit training samples (default: use all)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model")
    
    args = parser.parse_args()
    
    # Setup paths
    resolution = utils.get_resolution(args.resolution)
    dataset_dir = Path("nn/datasets")  # Resolution-independent datasets
    model_dir = Path(f"nn/models/{args.resolution}")  # Keep models separate by resolution
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists
    train_path = dataset_dir / config.COORD_FILE_TEMPLATE.format(split="train")
    val_path = dataset_dir / config.COORD_FILE_TEMPLATE.format(split="val")
    
    if not train_path.exists():
        print(f"Error: Training dataset not found at {train_path}")
        print(f"Run 'python nn/data.py --samples 10000' first")
        return
    
    # Create data loaders
    print(f"Loading datasets from {dataset_dir}")
    train_loader = DataLoader(str(train_path), args.batch_size, resolution, shuffle=True)
    
    # Limit samples if requested
    if args.samples and args.samples < train_loader.n_samples:
        train_loader.n_samples = args.samples
        train_loader.indices = train_loader.indices[:args.samples]
        print(f"Using {args.samples} / {len(train_loader.x)} available training samples")
    else:
        print(f"Using all {train_loader.n_samples} training samples")
    
    val_loader = None
    if val_path.exists():
        val_loader = DataLoader(str(val_path), args.batch_size, resolution, shuffle=False)
        print(f"Loaded {val_loader.n_samples} validation samples")
    
    # Create model
    model = MandelbrotCNN()
    
    # Load pretrained weights if provided
    start_epoch = 0
    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}")
        model.load_weights(args.pretrained)
        
    # Resume from checkpoint
    elif args.resume:
        checkpoints = list(model_dir.glob("model_epoch_*.npz"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            print(f"Resuming from {latest}")
            model.load_weights(str(latest))
            # Load metadata
            meta_path = latest.with_suffix('.json')
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    start_epoch = meta['epoch'] + 1
            else:
                start_epoch = 0
    
    # Setup optimizer and loss
    optimizer = optim.Adam(learning_rate=args.lr)
    loss_fn = MandelbrotLoss()
    
    # Training loop with early stopping
    print(f"\nStarting training on {resolution[0]}x{resolution[1]} images")
    # Count parameters using tree_flatten
    param_count = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model parameters: {param_count:,}")
    
    best_val_loss = float('inf')
    train_history = []
    patience_counter = 0
    initial_lr = optimizer.learning_rate
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        start_time = time.time()
        train_loss = train_epoch(model, optimizer, train_loader, loss_fn, epoch)
        epoch_time = time.time() - start_time
        
        # Evaluate
        metrics = {'epoch': epoch, 'train_loss': train_loss}
        
        if val_loader:
            val_metrics = evaluate(model, val_loader, loss_fn)
            metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, "
                  f"val_coord_mae={val_metrics['coord_mae']:.4f}, "
                  f"val_zoom_mae={val_metrics['zoom_mae']:.4f}, "
                  f"time={epoch_time:.1f}s")
            
            # Save best model and check early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                best_path = model_dir / config.BEST_MODEL_NAME
                # Save model weights
                model.save_weights(str(best_path))
                # Save metadata separately
                meta_path = model_dir / "best_model_meta.json"
                with open(meta_path, 'w') as f:
                    json.dump({'epoch': epoch, **metrics}, f, indent=2)
                print(f"  Saved best model to {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break
            
            # Learning rate scheduling (reduce LR on plateau)
            if patience_counter > 0 and patience_counter % 5 == 0:
                new_lr = optimizer.learning_rate * 0.5
                optimizer.learning_rate = new_lr
                print(f"  Reduced learning rate to {new_lr:.2e}")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, time={epoch_time:.1f}s")
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            checkpoint_path = model_dir / config.MODEL_CHECKPOINT_TEMPLATE.format(epoch=epoch)
            # Save model weights
            model.save_weights(str(checkpoint_path))
            # Save metadata
            meta_path = checkpoint_path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump({'epoch': epoch, **metrics}, f, indent=2)
            print(f"  Saved checkpoint to {checkpoint_path}")
        
        train_history.append(metrics)
    
    # Save training history
    history_path = model_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {model_dir}")


if __name__ == "__main__":
    main()