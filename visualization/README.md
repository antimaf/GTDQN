# Visualization

This directory contains the visualization tools and web interface for the GT-DQN poker agent.

## Components

### Web Interface
- `templates/`: HTML templates for the web interface
  - `index.html`: Main game interface
  - `styles.css`: CSS styling
- `static/`: Static assets (images, JavaScript)

### Features

1. **Game Visualization**
   - Real-time poker table display
   - Card visualization
   - Chip stack representation
   - Action buttons

2. **Training Metrics**
   - Learning curves
   - Win rate plots
   - Nash equilibrium convergence
   - Network statistics

3. **Interactive Play**
   - Human vs AI mode
   - AI vs AI simulation
   - Hand history replay

## Usage

To start the visualization server:
```python
from visualization.server import VisualizationServer

server = VisualizationServer(
    host="localhost",
    port=8080,
    model_path="checkpoints/latest.pth"
)

server.start()
```

Then open `http://localhost:8080` in your browser.

## Customization

The visualization can be customized through:
- `templates/styles.css`: Visual styling
- `config.json`: Server and display settings
- `static/images/`: Custom card and chip images

## Requirements
- Flask
- Plotly
- WebSocket-client
- jQuery
