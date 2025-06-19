# JoJo Face Stylization

This project provides real-time and image-based face stylization in the style of JoJo's Bizarre Adventure. It includes:
- A Flask API for image transformation
- A web frontend for easy interaction
- A desktop GUI for real-time webcam stylization

## Features
- Transform images into anime-style faces using deep learning
- Real-time webcam stylization (desktop GUI)
- REST API for programmatic access
- Web interface for easy upload and visualization

## Requirements
- Python 3.8+
- See `requirements.txt` for Python dependencies
- For GUI mode: `tkinter` (usually pre-installed, but may require `sudo apt-get install python3-tk` on Linux)
- Model weights: `weight.pth` (included)

## Installation
1. Clone the repository and navigate to the project folder:
   ```bash
   git clone <repo-url>
   cd face-style
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API Server
- Start the Flask API:
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:5000`.


## Using the Web Frontend
1. Open `index.html` in your browser (double-click or use a local server).
2. Upload an image and click "Transform Image" to see the result.
3. Use the real-time video section to stylize webcam input (requires API server running).


## Running the Desktop GUI
1. Make sure you have `tkinter` installed.
2. Run:
   ```bash
   python realtime_main.py
   ```
3. Use the GUI to upload images or start real-time webcam stylization.

## Test Images
Sample images are provided in the `test_img/` directory for quick testing.

## Slides and Report
- Project slides and report are available in the `slides/` directory.
