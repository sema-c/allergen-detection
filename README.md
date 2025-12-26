# Allergen Detection System

Deep learning-based allergen detection system for food images using YOLOv8.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project implements an object detection system for identifying allergens in food images. Using YOLOv8, the model can detect five different allergen categories in real-time.

### Detected Allergens
- Dairy Products
- Gluten
- Egg
- Seafood
- Nuts

## Model Performance

### General Metrics
- **Architecture**: YOLOv8n (Nano)
- **Input Size**: 416x416
- **Training Epochs**: 50
- **Dataset**: UECFOOD-256 (15,805 images)
- **Overall mAP@0.5**: 61.1%
- **Overall mAP@0.5-0.95**: 45.8%

### Class-wise Performance

| Allergen | mAP@0.5 | mAP@0.5-0.95 | Precision | Recall | Samples |
|----------|---------|--------------|-----------|--------|---------|
| Gluten | 82.1% | 63.6% | 68.6% | 87.1% | 1,757 |
| Seafood | 69.2% | 47.6% | 62.6% | 74.8% | 654 |
| Dairy | 66.0% | 51.7% | 55.6% | 78.4% | 533 |
| Egg | 53.9% | 40.0% | 49.6% | 69.4% | 589 |
| Nuts | 34.5% | 26.3% | 39.0% | 56.2% | 73 |

## Features

- Real-time allergen detection
- Multi-class object detection (5 allergen categories)
- Web-based interface (Gradio)
- Webcam support
- Heat map visualization
- Batch prediction capability
- Comprehensive performance metrics

## Installation

### Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. Clone the repository
```bash
git clone https://github.com/sema-c/allergen-detection.git
cd allergen-detection
```

2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download model weights

The trained model file (best.pt, ~6MB) is hosted separately due to size constraints:

**[Download Model (best.pt - 6MB)](https://drive.google.com/file/d/1-sP7pEr0ZgFj2L0-25WVUluogDiqgM1g/view?usp=sharing)**

Place the downloaded file in the `models/` directory:
```bash
# Linux/Mac
mv ~/Downloads/best.pt models/

# Windows
move Downloads\best.pt models\
```

## Usage

### Web Interface (Gradio)
```bash
python app/gradio_app.py
```
The interface will automatically open at `http://localhost:7860`

### Jupyter Notebook
```bash
jupyter notebook notebooks/training.ipynb
```

### Python API
```python
from ultralytics import YOLO

# Load model
model = YOLO('models/best.pt')

# Run prediction
results = model.predict('path/to/image.jpg', conf=0.25)

# Display results
results[0].show()
```

## Project Structure
```
allergen-detection/
├── notebooks/              # Jupyter notebooks
│   ├── training.ipynb     # Model training
│   └── evaluation.ipynb   # Model evaluation
├── app/                    # Web interfaces
│   ├── gradio_app.py      # Gradio interface
│   └── streamlit_app.py   # Streamlit interface
├── models/                 # Trained models
│   └── best.pt            # Best model weights
├── results/                # Training results
│   ├── confusion_matrix.png
│   ├── results.png
│   └── training_curves.png
├── docs/                   # Documentation
│   └── REPORT.md          # Detailed project report
├── assets/                 # Images and media
│   └── demo.gif           # Demo visualization
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
├── LICENSE                # MIT License
└── README.md              # This file
```

## Methodology

### Dataset
- **Source**: UECFOOD-256
- **Total Images**: 31,397
- **Training Set**: 12,344 images
- **Validation Set**: 3,461 images
- **Annotations**: 18,097 allergen labels

### Training Configuration
- **Model**: YOLOv8n
- **Optimizer**: AdamW
- **Learning Rate**: 0.01
- **Batch Size**: 32
- **Epochs**: 50
- **Image Size**: 416x416
- **Augmentation**: Mosaic, MixUp, HSV, Flip

### Data Augmentation
- Random horizontal flip
- HSV color jittering
- Mosaic augmentation
- MixUp augmentation

## Results

### Strengths
- Gluten detection achieves highest performance at 82.1% mAP@0.5
- Average recall of 73.2% demonstrates strong detection capability
- Balanced performance across seafood and dairy categories

### Limitations
- Lower performance on nuts class (only 73 training samples)
- Moderate accuracy on egg detection
- Challenges with small object detection

## Citation

If you use this project in your research, please cite:
```bibtex
@misc{allergen_detection_2024,
  title={Allergen Detection System using YOLOv8},
  author={TÜBİTAK 2209-A Research Project},
  year={2024},
  howpublished={\url{https://github.com/sema-c/allergen-detection}}
}
```

## Disclaimer

**IMPORTANT**: This system is designed for research purposes only and should not be used for medical decision-making. Always consult healthcare professionals regarding food allergies and dietary restrictions. The model does not guarantee 100% accuracy and should undergo comprehensive testing before any critical applications.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [UECFOOD-256 Dataset](http://foodcam.mobi/dataset256.html) - Dataset provider
- [TÜBİTAK 2209-A Program](https://www.tubitak.gov.tr) - Research funding
- [Gradio](https://gradio.app/) - Web interface framework

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: Star this repository if you find it useful!
