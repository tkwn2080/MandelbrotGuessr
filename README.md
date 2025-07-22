# MandelbrotGuessr

A game where players guess locations in the Mandelbrot set, paired with a neural network that learns to play using reinforcement learning-style training.

## The Game (`index.html`)

MandelbrotGuessr is a browser-based game inspired by GeoGuessr. Players are shown a zoomed-in view of the Mandelbrot set and must click on a minimap to guess where in the fractal they are looking.

### Game Mechanics
- **5 rounds** per game, 60 seconds each
- **Main view**: Shows the target location at various zoom levels
- **Minimap**: Interactive map where players click to guess
  - Pan by dragging
  - Zoom with scroll wheel
  - Click to place guess
- **Scoring**: Based on distance between guess and actual location
  - Score = 10,000 � exp(-distance / 0.3)
  - Maximum: 10,000 points (perfect guess)
  - Minimum: 0.01 points (very far off)

### Key Features
- WebGL acceleration for smooth fractal rendering
- Swappable views (Tab key)
- Visual feedback showing guess accuracy
- Local high score tracking
- Responsive zoom animations

## Neural Network (`nn/`)

A convolutional neural network that learns to play MandelbrotGuessr using a unique reinforcement learning approach.

### Architecture (`model.py`)

**MandelbrotCNN**: Resolution-agnostic CNN with global average pooling
- Feature extraction: Conv blocks with batch normalization
- Output: 5 values for ScreenLoss mode
  1. View center X (normalized)
  2. View center Y (normalized)  
  3. Log10(zoom)
  4. Click X position (0-1)
  5. Click Y position (0-1)

### Loss Function: ScreenLoss

Instead of traditional supervised learning, the model optimizes game score directly:

```python
# Simulates human gameplay:
1. Model selects view center and zoom
2. Model clicks within that view (pixel-limited)
3. Score calculated from final distance
4. Loss = -mean(game_scores)
```

Key innovations:
- **Pixel discretization**: Simulates human clicking limitations (800�600 pixels)
- **View selection**: Model must choose where to look AND where to click
- **Zoom emerges naturally**: High zoom needed for precision due to pixel constraints

### Training (`train.py`)

- Uses MLX framework (optimized for Apple Silicon)
- Direct game score optimization (RL-style)
- Gradient clipping for stability
- Early stopping with patience

### Data Generation (`data.py`, `utils.py`)

- Uniform sampling across Mandelbrot set
- Full complex plane coverage (-2.5 to 1.0, -1.25 to 1.25)
- Quality filtering using entropy and edge detection
- Multiple zoom levels (10� to 100,000�)

### Evaluation (`eval.py`)

Comprehensive evaluation including:
- Game score metrics
- Strategy analysis (zoom distribution, click patterns)
- Gaming behavior detection
- Visualization of model decisions

## Key Insights

1. **Reinforcement Learning Approach**: The model optimizes game score directly rather than minimizing coordinate prediction error
2. **Emergent Behavior**: Zoom strategy emerges from pixel constraints, not explicit programming
3. **Human-like Constraints**: Pixel discretization forces the model to face the same precision/coverage tradeoffs as humans
4. **Imperfect Centering**: Model learns that perfect view centering doesn't improve score, focusing effort on final click accuracy

## Setup

### Game
Simply open `index.html` in a modern web browser (desktop only).

### Neural Network
```bash
# Install dependencies
pip install -r requirements.txt  # or use uv

# Generate training data
python nn/data.py --samples 10000

# Train model
python nn/train.py --loss screen --epochs 50

# Evaluate
python nn/eval.py --model nn/models/320x240/best_model.npz
```

## Future Directions

- **Spatial Attention**: Adding attention mechanisms to help the model focus on distinctive fractal patterns
- **Recurrent Search**: Implementing multi-step refinement where the model can:
  - Make an initial guess and view that location
  - Store visited locations in memory
  - Iteratively refine its guess based on visual feedback
  - Select a final answer after multiple exploration steps
  - This would more closely mimic human search strategies
- Higher resolution training
- Deployment of trained model in the web game
- Competitive play between human and AI