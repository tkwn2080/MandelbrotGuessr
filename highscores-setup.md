# Setting Up High Scores with GitHub Gist

## Current Implementation
The game currently uses localStorage for high scores, which works but is only local to each browser.

## Security Warning
**IMPORTANT**: GitHub Gists are public and anyone with a token can write to them. This means:
- Other people could write fake scores to your gist
- Your GitHub token would be exposed if you put it in client-side code
- This is why the game uses localStorage by default

## To Enable Cloud High Scores (Read-Only):

### 1. Create a GitHub Gist
1. Go to https://gist.github.com
2. Create a new public gist named `mandelbrot-highscores.json`
3. Initialize it with this content:
```json
[]
```
4. Save the gist and copy its ID from the URL

### 2. Update the Game Code
Replace `YOUR_GIST_ID` in index.html with your actual Gist ID.

### 3. For Read-Only High Scores
- No GitHub token needed
- Scores can be viewed but not submitted online
- You would manually update the gist with legitimate scores

## Secure Alternatives:

### Option 1: Firebase (Recommended)
- Free tier is generous
- Works directly from client-side
- Can add authentication rules
- Real-time updates
- Much more secure than Gists

### Option 2: Vercel/Netlify Functions
- Create a serverless function to handle submissions
- Store GitHub token server-side
- Validate scores before saving
- More complex but fully secure

### Option 3: Keep it Local
- Current localStorage solution works fine
- No security concerns
- Players can compete locally
- Simple and effective for a GitHub Pages site

## Current Features
- 5 rounds per game
- 60-second timer per round
- Score based on accuracy and zoom level
- Accounts for Mandelbrot set symmetry
- Local high score storage
- Pan mode with hand icon (shift+drag or toggle)