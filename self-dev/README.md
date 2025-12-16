# Meta Mynd - Developer Version

The developer version of MYND, refactored into separate files for better maintainability.

## Directory Structure

```
self-dev/
├── index.html          # Main entry point
├── manifest.json       # PWA manifest
├── mynd-app-data.js    # Embedded mind map data
├── README.md           # This file
├── css/
│   └── styles.css      # All CSS styles (~192KB)
├── js/
│   ├── app.js          # Main application logic (~1.3MB)
│   ├── config.js       # Configuration
│   └── goal-system.js  # Goal tracking system
└── icons/
    ├── icon-192.png
    └── icon-512.png
```

## How to Run

The application uses ES6 modules, which require a local server due to CORS restrictions.

### Option 1: Python HTTP Server
```bash
cd self-dev
python3 -m http.server 8000
```
Then open: http://localhost:8000

### Option 2: Node.js HTTP Server
```bash
cd self-dev
npx http-server -p 8000
```
Then open: http://localhost:8000

### Option 3: VS Code Live Server
If using VS Code, install the "Live Server" extension and right-click on `index.html` → "Open with Live Server"

## Dependencies

External dependencies are loaded via CDN:
- **Three.js** v0.160.0 - 3D visualization
- **Supabase** v2 - Authentication & cloud sync
- **TensorFlow.js** v4.17.0 - Machine learning (lazy loaded)
- **Universal Sentence Encoder** v1.3.3 - Text embeddings (lazy loaded)
- **Google Fonts** - Inter, Space Grotesk, JetBrains Mono

## File Descriptions

### CSS (css/styles.css)
Contains all styles including:
- Theme system (6 themes: sandstone, coral, ember, frost, obsidian)
- Base styles & reset
- Component styles (header, panels, modals)
- Mobile responsive styles
- Animation keyframes

### JavaScript (js/app.js)
Contains all application logic including:
- Supabase client initialization & auth
- TensorFlow lazy loader
- Animation controller
- Store class (mind map data management)
- PersonalNeuralNet (AI/ML system)
- SemanticEngine (text embeddings & similarity)
- MetaLearner (learning strategies)
- Three.js scene management
- UI handlers & event listeners
- Mobile bottom sheet
- Voice/chat functionality

### Configuration (js/config.js)
Contains:
- API endpoints
- Supabase credentials (placeholder values)
- OpenAI API configuration
- Default settings
- Theme colors

### Goal System (js/goal-system.js)
Contains:
- GoalTracker class
- Goal creation & management
- Progress tracking
- Goal completion animations

## Key Features

- **3D Mind Map Visualization** - Three.js powered
- **AI Suggestions** - TensorFlow.js neural network
- **Voice Input** - Speech recognition
- **Cloud Sync** - Supabase real-time sync
- **6 Color Themes** - Light and dark options
- **Mobile Responsive** - Touch-optimized UI
- **Self Developer Mode** - Local development features

## Version

**Meta Mynd V5.0.1** (Developer Version)

## Notes

- The JavaScript is kept as a single `app.js` file to preserve all functionality and avoid circular dependency issues
- All external scripts are loaded via CDN for simplicity
- The application requires a modern browser with ES6 module support
