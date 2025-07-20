# MandelbrotGuessr

A GeoGuessr-style game where players guess their location within the Mandelbrot set! Players are shown a zoomed-in view of the fractal and must click on a minimap to guess where they are.

## Features

- 🎮 Interactive gameplay with scoring system
- 🗺️ Clickable minimap for making guesses
- 🎯 Difficulty levels from Easy (zoomed out) to Extreme (deep zoom)
- 🎨 Beautiful color-coded Mandelbrot visualizations
- 📊 Automatic location generation based on entropy and edge detection
- 🚀 Runs entirely in the browser - perfect for GitHub Pages

## Setup Instructions

### 1. Generate Interesting Locations

First, run the Python script to generate a collection of interesting Mandelbrot locations:

```bash
# Install required dependencies
pip install numpy scipy matplotlib

# Generate locations
python mandelbrot_location_generator.py
```

This will create a `mandelbrot_locations.json` file with 100 interesting locations.

### 2. Convert Locations for Web

Convert the JSON locations to a JavaScript file:

```bash
python location_converter.py
```

This creates `locations.js` that the web game can load.

### 3. Deploy to GitHub Pages

1. Create a new GitHub repository
2. Add these files to your repository:
   - `index.html` (the game file)
   - `locations.js` (generated locations)

3. Enable GitHub Pages:
   - Go to Settings → Pages
   - Set source to "Deploy from a branch"
   - Select "main" branch and "/ (root)" folder
   - Save

4. Your game will be available at:
   ```
   https://[your-username].github.io/[repository-name]/
   ```

## File Structure

```
mandelbrot-guessr/
├── index.html                    # Main game file
├── locations.js                  # Generated locations (created by converter)
├── mandelbrot_location_generator.py  # Python script to find locations
├── location_converter.py         # Converts JSON to JS
├── mandelbrot_locations.json    # Raw location data (generated)
└── README.md                    # This file
```

## How the Location Generation Works

The Python script uses several techniques to find interesting locations:

1. **Entropy Calculation**: Measures the complexity of the image using Shannon entropy
2. **Edge Detection**: Uses Sobel edge detection to find areas with lots of detail
3. **Multi-scale Search**: Searches at different zoom levels from 1x to 10,000x
4. **Interest Scoring**: Combines entropy and edge density to score locations

## Game Mechanics

- **Scoring**: Up to 5,000 points per round based on accuracy
- **Distance**: Calculated in Mandelbrot coordinate space
- **Difficulty**: Based on zoom level:
  - Easy: 1-2x zoom
  - Medium: 2-10x zoom
  - Hard: 10-100x zoom
  - Very Hard: 100-1000x zoom
  - Extreme: 1000x+ zoom

## Customization

### Adding More Locations

Edit `mandelbrot_location_generator.py` to:
- Change the number of locations generated
- Adjust the search regions
- Modify the interest thresholds

### Changing Colors

In `index.html`, modify the `hslToRgb` function call:
```javascript
const hue = (iteration / maxIterations) * 360;
const rgb = hslToRgb(hue, 100, 50);  // Adjust saturation and lightness
```

### Adjusting Difficulty

Modify the scoring formula in `makeGuess()`:
```javascript
const score = Math.max(0, Math.round(maxScore * Math.exp(-distance * currentLocation.zoom / 10)));
```

## Performance Tips

- The game renders at 800x600 resolution by default
- Minimap uses reduced iterations (50) for faster rendering
- Deep zoom locations may take a moment to render

## Browser Compatibility

Works in all modern browsers that support:
- HTML5 Canvas
- ES6 JavaScript
- Async/await

## Future Enhancements

- [ ] Multiplayer mode
- [ ] Daily challenges
- [ ] Leaderboard system
- [ ] More color schemes
- [ ] Progressive zoom animation
- [ ] Hints system
- [ ] Mobile touch controls

## License

This project is open source and available under the MIT License.

## Credits

Inspired by GeoGuessr and the mathematical beauty of the Mandelbrot set!
