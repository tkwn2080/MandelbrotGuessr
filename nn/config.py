"""Configuration constants for Mandelbrot location predictor."""

# Mandelbrot generation parameters
MAX_ITERATIONS = 256
BAILOUT_RADIUS = 2.0

# Data generation parameters
# Log10 zoom ranges
ZOOM_RANGE_TRAIN = (3.0, 4.0)  # 1,000 to 10,000
ZOOM_RANGE_TEST1 = (4.0, 4.3)   # 10,000 to ~20,000
ZOOM_RANGE_TEST2 = (4.3, 5.0)   # ~20,000 to 100,000

# Coordinate constraints
Y_MIN = -1.25  # Full complex plane (both upper and lower half)
X_MIN = -2.5
X_MAX = 1.0
Y_MAX = 1.25

# Default resolution (can be overridden via CLI)
DEFAULT_RESOLUTION = (320, 240)  # 4:3 aspect ratio
SUPPORTED_RESOLUTIONS = {
    # 4:3 aspect ratio (matches web game)
    "320x240": (320, 240),    # Fast training
    "640x480": (640, 480),    # Medium quality
    "800x600": (800, 600),    # High quality (matches web game exactly)
    # 1:1 aspect ratio (legacy/special use)
    "256": (256, 256),
    "512": (512, 512),
}

# Training parameters
BATCH_SIZE = 16  # Smaller batch size for faster iteration
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 100
GRADIENT_CLIP_NORM = 1.0  # For gradient clipping
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for N epochs

# Loss function weights
COORDINATE_LOSS_WEIGHT = 1.0
ZOOM_LOSS_WEIGHT = 0.5  # Scaled down since log zoom has smaller range

# Model architecture (simplified)
FEATURE_DIMS = [32, 64, 128]  # Reduced channel progression
ADAPTIVE_POOL_SIZE = (4, 4)  # Size before dense layers
HIDDEN_DIM = 128  # Smaller hidden layer

# Data split ratios
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation

# File naming conventions
COORD_FILE_TEMPLATE = "coords_{split}.npz"
MODEL_CHECKPOINT_TEMPLATE = "model_epoch_{epoch:04d}.npz"
BEST_MODEL_NAME = "best_model.npz"

# Logging and display
LOG_INTERVAL = 10  # Log every N batches
SAVE_INTERVAL = 10  # Save checkpoint every N epochs
VISUALIZATION_SAMPLES = 16  # Number of samples to visualize

# MLX specific settings
MLX_DTYPE = "float32"