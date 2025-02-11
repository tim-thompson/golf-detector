# Golf Ball Tracking Application

A high-performance application for tracking and tracing golf ball trajectories in video footage. This application uses computer vision and machine learning techniques to accurately detect and track golf balls, creating smooth and professional-looking trajectory visualizations.

## Features

- Accurate golf ball detection and tracking
- Smooth trajectory visualization with anti-aliasing
- Support for various lighting conditions and backgrounds
- GPU acceleration support (when available)
- Real-time or near-real-time processing capabilities
- Professional-looking output with customizable trajectory appearance

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- CUDA-capable GPU (optional, for GPU acceleration)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/golf-detector.git
cd golf-detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application using the following command:

```bash
python src/main.py --input path/to/your/video.mp4 --output path/to/output.mp4
```

### Command-line Arguments

- `--input`: Path to the input video file (required)
- `--output`: Path to save the output video (default: output.mp4)
- `--use-gpu`: Enable GPU acceleration if available (optional)
- `--trace-color`: Color of the trajectory trace (default: white)
- `--trace-thickness`: Thickness of the trajectory trace (default: 2)

Example with all options:
```bash
python src/main.py --input golf_swing.mp4 --output tracked_swing.mp4 --use-gpu --trace-color blue --trace-thickness 3
```

## Performance Optimization

The application is optimized for performance in several ways:

1. GPU acceleration when available
2. Efficient frame processing using OpenCV
3. Kalman filtering for smooth trajectory prediction
4. Background subtraction for improved detection
5. Multi-threaded processing where applicable

## Limitations

- Best results are achieved with videos shot from a stable camera position
- Ball detection may be affected by very poor lighting conditions
- Processing speed depends on hardware capabilities and video resolution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 