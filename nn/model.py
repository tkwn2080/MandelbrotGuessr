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
            nn.Linear(hidden_dim, 5)  # Output: [view_x, view_y, log10_zoom, click_x, click_y]
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


class ScreenLoss(nn.Module):
    """
    Loss function that mimics human gameplay with view selection and pixel clicking.
    
    The NN predicts:
    - view_center_x, view_center_y: Where to center the zoomed view (normalized)
    - log_zoom: Zoom level for the view
    - click_x, click_y: Where to click within that view (0-1 normalized)
    
    This forces the NN to:
    1. Navigate to the right region (view center + zoom)
    2. Click precisely within that region (pixel-limited)
    """
    
    def __init__(
        self,
        minimap_width: int = 800,
        minimap_height: int = 600,
        max_score: float = 10000.0,
        decay_factor: float = 0.3,  # Increased from 0.05 for better gradient flow
        zoom_regularization: float = 0.01,
        min_zoom: float = 1.0,
        max_zoom: float = 100000.0
    ):
        super().__init__()
        self.minimap_width = minimap_width
        self.minimap_height = minimap_height
        self.max_score = max_score
        self.decay_factor = decay_factor
        self.zoom_regularization = zoom_regularization
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        
        # Full Mandelbrot view parameters
        self.full_x_range = 3.5
        self.full_y_range = 2.5
        
    def view_and_click_to_coords(self, view_x, view_y, zoom, click_x, click_y):
        """
        Convert view center and click position to final Mandelbrot coordinates.
        
        Args:
            view_x, view_y: Normalized view center coordinates
            zoom: Zoom level (not log)
            click_x, click_y: Normalized click position (0-1) within the view
            
        Returns:
            Final Mandelbrot coordinates after pixel discretization
        """
        # Denormalize view center
        view_center_x = view_x * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
        view_center_y = view_y * config.Y_MAX
        
        # Calculate visible range at this zoom
        view_width = self.full_x_range / zoom
        view_height = self.full_y_range / zoom
        
        # Convert click position to pixels
        pixel_x = mx.round(click_x * self.minimap_width)
        pixel_y = mx.round(click_y * self.minimap_height)
        
        # Convert pixels to offset from view center
        # pixel (400, 300) = center of view (0 offset)
        offset_x = (pixel_x - self.minimap_width/2) * view_width / self.minimap_width
        offset_y = (pixel_y - self.minimap_height/2) * view_height / self.minimap_height
        
        # Final coordinates
        final_x = view_center_x + offset_x
        final_y = view_center_y + offset_y
        
        return final_x, final_y
        
    def __call__(self, pred, target, return_components=False):
        """
        Calculate game score loss with view center and click position predictions.
        
        Args:
            pred: [batch, 5] containing [view_x, view_y, log10_zoom, click_x, click_y]
            target: [batch, 3] containing [x_norm, y_norm, log10_zoom] of true location
            
        Returns:
            Loss value (negative mean game score)
        """
        # Extract predictions
        pred_view_x = pred[:, 0]    # Normalized view center x
        pred_view_y = pred[:, 1]    # Normalized view center y
        pred_log_zoom = pred[:, 2]  # Log10 zoom
        pred_click_x = pred[:, 3]   # Click position x (0-1)
        pred_click_y = pred[:, 4]   # Click position y (0-1)
        
        # Convert log zoom to linear
        pred_zoom = mx.power(10, pred_log_zoom)
        pred_zoom = mx.clip(pred_zoom, self.min_zoom, self.max_zoom)
        
        # Convert view + click to final coordinates
        pred_x, pred_y = self.view_and_click_to_coords(
            pred_view_x, pred_view_y, pred_zoom, pred_click_x, pred_click_y
        )
        
        # Denormalize true coordinates
        true_x = target[:, 0] * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
        true_y = target[:, 1] * config.Y_MAX
        true_zoom = mx.power(10, target[:, 2])
        
        # Calculate distance in Mandelbrot space
        distance = mx.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        
        # Simple scoring - pixel discretization makes zoom necessary
        score_ratio = mx.exp(-distance / self.decay_factor)
        game_scores = self.max_score * score_ratio
        
        # Loss is negative score
        score_loss = -mx.mean(game_scores)
        
        # Zoom regularization
        zoom_reg = self.zoom_regularization * mx.mean(pred_log_zoom**2)
        
        # Additional regularization: penalize extreme click positions
        # Encourage clicking near center when possible
        click_center_reg = 0.001 * mx.mean((pred_click_x - 0.5)**2 + (pred_click_y - 0.5)**2)
        
        total_loss = score_loss + zoom_reg + click_center_reg
        
        if return_components:
            # Calculate view center accuracy
            view_center_x_denorm = pred_view_x * ((config.X_MAX - config.X_MIN) / 2) + ((config.X_MIN + config.X_MAX) / 2)
            view_center_y_denorm = pred_view_y * config.Y_MAX
            view_error = mx.sqrt((view_center_x_denorm - true_x)**2 + (view_center_y_denorm - true_y)**2)
            
            return {
                'total_loss': total_loss,
                'score_loss': score_loss,
                'zoom_reg': zoom_reg,
                'click_reg': click_center_reg,
                'mean_score': mx.mean(game_scores),
                'mean_distance': mx.mean(distance),
                'mean_view_error': mx.mean(view_error),
                'mean_pred_zoom': mx.mean(pred_zoom)
            }
        
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