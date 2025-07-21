"""Resolution-agnostic CNN model for Mandelbrot coordinate prediction."""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple
import config


class ConvBlock(nn.Module):
    """Convolutional block with conv -> batchnorm -> relu."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()
        
    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Removed ResBlock - keeping architecture simple


class MandelbrotCNN(nn.Module):
    """
    Resolution-agnostic CNN for Mandelbrot coordinate prediction.
    
    Uses global average pooling to handle any input resolution.
    Predicts (x, y, log10_zoom) coordinates.
    """
    
    def __init__(
        self,
        feature_dims: List[int] = config.FEATURE_DIMS,
        hidden_dim: int = config.HIDDEN_DIM
    ):
        super().__init__()
        
        # Input is single channel (grayscale)
        in_channels = 1
        
        # Feature extraction layers - simplified architecture
        self.features = []
        for out_channels in feature_dims:
            self.features.append(ConvBlock(in_channels, out_channels))
            # Downsample after each conv block
            self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        # We'll use global average pooling instead of adaptive pooling
        # This reduces each feature map to a single value
        # Much simpler and works with any input resolution
        
        # Output size after global pooling is just the number of channels
        pool_output_size = feature_dims[-1]
        
        # Simplified prediction head with less dropout
        self.predictor = [
            nn.Linear(pool_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout
            nn.Linear(hidden_dim, 3)  # Output: [x, y, log10_zoom]
        ]
        
    def __call__(self, x):
        # Add channel dimension if needed (MLX uses NHWC format)
        if len(x.shape) == 3:
            x = mx.expand_dims(x, axis=-1)  # Add channel as last dimension
        
        # Feature extraction
        for layer in self.features:
            x = layer(x)
        
        # Global average pooling
        # For NHWC format: average over H and W dimensions (axes 1 and 2)
        x = mx.mean(x, axis=(1, 2))  # Shape: [B, C]
        
        # Prediction
        for layer in self.predictor:
            x = layer(x)
        
        return x


class MandelbrotLoss(nn.Module):
    """
    Combined loss function for coordinate prediction.
    
    Uses Huber loss for robustness and balances coordinate vs zoom losses.
    """
    
    def __init__(
        self,
        coord_weight: float = config.COORDINATE_LOSS_WEIGHT,
        zoom_weight: float = config.ZOOM_LOSS_WEIGHT,
        huber_delta: float = 1.0
    ):
        super().__init__()
        self.coord_weight = coord_weight
        self.zoom_weight = zoom_weight
        self.huber_delta = huber_delta
        
    def huber_loss(self, pred, target):
        """Use MLX's built-in Huber loss."""
        return nn.losses.huber_loss(pred, target, delta=self.huber_delta, reduction='none')
        
    def __call__(self, pred, target):
        """
        Calculate combined loss.
        
        Args:
            pred: Predictions [batch, 3] containing [x, y, log10_zoom]
            target: Targets [batch, 3] containing [x, y, log10_zoom]
            
        Returns:
            Scalar loss value
        """
        # Separate coordinates and zoom
        coord_pred = pred[:, :2]  # x, y
        zoom_pred = pred[:, 2:3]   # log10_zoom
        
        coord_target = target[:, :2]
        zoom_target = target[:, 2:3]
        
        # Calculate separate losses
        coord_loss = mx.mean(self.huber_loss(coord_pred, coord_target))
        zoom_loss = mx.mean(self.huber_loss(zoom_pred, zoom_target))
        
        # Combine with weights
        total_loss = self.coord_weight * coord_loss + self.zoom_weight * zoom_loss
        
        return total_loss


def create_model(resolution: Tuple[int, int] = None) -> MandelbrotCNN:
    """
    Create a MandelbrotCNN model.
    
    Args:
        resolution: Optional input resolution for initialization
        
    Returns:
        MandelbrotCNN model
    """
    return MandelbrotCNN()


def test_model():
    """Test model with different resolutions."""
    model = create_model()
    
    # Test different resolutions
    for res_name, (w, h) in config.SUPPORTED_RESOLUTIONS.items():
        # Create dummy input
        batch_size = 4
        x = mx.random.uniform(shape=(batch_size, h, w))
        
        # Forward pass
        output = model(x)
        
        print(f"Resolution {res_name} ({w}x{h}):")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output sample: {output[0]}")
        print()


if __name__ == "__main__":
    test_model()