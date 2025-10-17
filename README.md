# Animal Injury Detection Assisted System

A comprehensive end-to-end machine learning system for identifying and classifying animal diseases from images using deep learning techniques.

## 🌟 Features

- **Multiple Model Architectures**: Custom CNNs, ResNet, EfficientNet, VGG, Inception, and DenseNet
- **Data Preprocessing**: Advanced image preprocessing and augmentation pipelines
- **Transfer Learning**: Pre-trained models fine-tuned for animal disease classification
- **Web Interface**: User-friendly web application for image upload and prediction
- **REST API**: FastAPI-based API for integration with other systems
- **Batch Processing**: Support for multiple image analysis
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualization
- **Model Deployment**: Easy model serialization and deployment utilities

🌈 Grad-CAM Explainability & Visualization
Integrated Grad-CAM Visualization: Upload images and visualize attention heatmaps to see which regions influenced the model’s decision.

Dedicated Grad-CAM Web Interface: Access /gradcam route and template for interactive explanations.

Programmatic Access: Use the new /predict_gradcam API for JSON visualizations with base64-encoded heatmaps.

Diagnostic Scripts: Tools like debug_gradcam.py, analyze_model.py, and test_gradcam.py for import validation, model architecture compatibility, and end-to-end Grad-CAM testing.

Educational & Clinical Value: Helps users trust and interpret predictions, understand AI reasoning, and validates veterinary results.

Robustness: Graceful fallback and clear errors if Grad-CAM is unavailable, with auto-detected layers and supported transfer learning models.

## Concise Project Structure

At a glance — where to find data and main components:

```
c:\Users\priti\Projects\AIDAS\
├── data/                        # Datasets
│   ├── raw/                     # Original images (per-class folders e.g. healthy/)
│   ├── sample/                  # Small sample dataset for quick tests
│   ├── enhanced_dataset/        # Preprocessed dataset used by enhanced training
│   └── ultimate_dataset/        # Dataset assembled by ultimate training script
├── models/                      # Saved model files and results JSON
├── src/                         # Application source (config, inference, gradcam, etc.)
│   ├── gradcam.py               # Main Grad-CAM implementation
│   ├── inference.py             # Enhanced with Grad-CAM support
│   └── ...                      # Other modules
├── templates/                   # Flask HTML templates (used by enhanced_app.py)
│   └── gradcam_index.html       # Dedicated Grad-CAM interface
├── static/                      # Static assets for the web UI
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Test scripts
│   ├── test_gradcam.py          # Comprehensive test suite for Grad-CAM
├── scripts/ or root scripts     # Training scripts (enhanced_90plus_train.py, ultimate_train.py, etc.)
├── enhanced_app.py              # Flask app with Grad-CAM routes and initialization
├── debug_gradcam.py             # Diagnostic tool for Grad-CAM import/troubleshooting
├── analyze_model.py             # Model architecture analyzer (for Grad-CAM layers)
├── LICENSE
├── README.md
└── requirements.txt

```

Notes:
- Put raw images under data/raw/<class_name>/ (e.g. data/raw/healthy/).
- Trained models and result files live in the models/ directory.
- Web UI is in enhanced_app.py (root) and uses templates/ and static/.
- Source modules live in src/ for import by scripts and web app.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd animal-disease-classifier

# Create virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Create your dataset structure:

```bash
data/raw/
├── healthy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── bacterial_infection/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── viral_infection/
│   └── ...
└── fungal_infection/
    └── ...
```

### 3. Preprocess Data

```python
from src.data_preprocessing import DataPreprocessor
from src.config import Config

config = Config()
preprocessor = DataPreprocessor(config)

# Organize dataset
preprocessor.organize_dataset_from_folder('data/raw', 'data/processed')

# Create train/validation/test splits
preprocessor.create_train_val_test_split('data/processed')
```

### 4. Train a Model

```python
from src.train import train

# Train with EfficientNet-B0 (recommended)
history = train('efficientnet_b0')

# Or try other models
# history = train('resnet_50')
# history = train('simple_cnn')
```

### 5. Evaluate Model

```python
from src.evaluation import evaluate_trained_model
from src.config import Config

config = Config()
results = evaluate_trained_model(config.BEST_MODEL_PATH, config.TEST_DATA_DIR)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### 6. Make Predictions

```python
from src.inference import DiseasePredictor
from src.config import Config

config = Config()
predictor = DiseasePredictor(config.BEST_MODEL_PATH, config)

# Single image prediction
result = predictor.get_prediction_explanation('path/to/image.jpg')
print(f"Prediction: {result['top_prediction']['class']}")
print(f"Confidence: {result['top_prediction']['confidence']:.4f}")
```

### 7. Run Web Application

```bash
cd src
python app.py
```

Visit `http://localhost:5000` to access the web interface.

### 8. GradCam implementation

What is Grad-CAM?
Grad-CAM (Gradient-weighted Class Activation Mapping) generates visual explanations for the decisions made by convolutional neural networks. It highlights the important regions in an image that influenced the model’s predictions.

How Grad-CAM Works in This Project
Initialization: Grad-CAM integration starts on app launch and auto-detects suitable layers for explanation.

User Flow: Upload images via the /gradcam web interface, trigger Grad-CAM, and receive visualization heatmaps.

API: New endpoints like /predict_gradcam for programmatic access, yielding JSON results with model predictions, heatmap (base64-encoded), and feature explanations.

Files and Modules:

src/gradcam.py: Core heatmap generator and integration utilities

src/inference.py: Optional Grad-CAM import and explanations

enhanced_app.py: Grad-CAM initialization, endpoints/routes

templates/gradcam_index.html: Dedicated Grad-CAM upload/visualization page

debug_gradcam.py, analyze_model.py, test_gradcam.py: Model compatibility diagnostics, layer analysis, and test suites

Example Usage (Web & API)
# Single image Grad-CAM prediction
from src.inference import DiseasePredictor
result = predictor.get_prediction_explanation('path/to/image.jpg')
heatmap = result['gradcam']['main_heatmap'] # base64 PNG for visualization

API endpoint:
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict_gradcam

Web Interface
Visit /gradcam for interactive Grad-CAM demonstrations

Upload images to view attention heatmaps highlighting model reasoning

Compare multiple-class visualizations

Utility Scripts
debug_gradcam.py: Test Grad-CAM imports, model compatibility, troubleshoot issues

analyze_model.py: Automatically locate layers for Grad-CAM in custom and transfer learning models

test_gradcam.py: End-to-end Grad-CAM test and performance validation

Error Handling & Robustness
Graceful fallback if Grad-CAM unavailable

Extensive compatibility checks and gradient validation

Fallback visualizations and user-friendly error messages

Educational & Clinical Impact
Builds trust by explaining AI predictions

Allows visual learning of disease patterns and model focus

Facilitates validation and adoption in veterinary/clinical contexts

## 🎯 Supported Disease Categories

The system supports classification of the following animal diseases:

- **Healthy**: Normal, disease-free animals
- **Bacterial Infection**: Bacterial-caused diseases
- **Viral Infection**: Viral-caused diseases  
- **Fungal Infection**: Fungal-caused diseases
- **Parasitic Infection**: Parasite-caused diseases
- **Nutritional Deficiency**: Nutrition-related health issues
- **Genetic Disorder**: Hereditary conditions

*Note: Modify `DISEASE_CATEGORIES` in `config.py` based on your specific dataset.*

## 🏗️ Model Architectures

### Available Models

| Model Type | Description | Use Case |
|------------|-------------|----------|
| `simple_cnn` | Basic CNN for baseline | Quick prototyping |
| `advanced_cnn` | Custom CNN with residual connections | Balanced performance |
| `resnet_50/101/152` | ResNet variants | High accuracy |
| `efficientnet_b0/b1/b2/b3` | EfficientNet variants | Best efficiency |
| `vgg16/vgg19` | VGG architectures | Transfer learning |
| `inception` | InceptionV3 | Feature diversity |
| `densenet` | DenseNet121 | Parameter efficiency |

### Creating a Model

```python
from src.models import ModelFactory

factory = ModelFactory()
model = factory.create_model(
    model_type='efficientnet_b0',
    num_classes=7,
    input_shape=(224, 224, 3)
)
```

## 📊 Training and Evaluation

### Training Configuration

Modify training parameters in `config.py`:

```python
class Config:
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
```

### Custom Training Script

```python
from src.models import ModelFactory
from src.data_preprocessing import DataPreprocessor
from src.config import Config

config = Config()
preprocessor = DataPreprocessor(config)

# Create datasets
train_ds = preprocessor.create_tensorflow_dataset(config.TRAIN_DATA_DIR, augment=True)
val_ds = preprocessor.create_tensorflow_dataset(config.VAL_DATA_DIR, augment=False)

# Create and compile model
factory = ModelFactory(config)
model = factory.create_model('efficientnet_b0', len(train_ds.class_names))

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config.EPOCHS,
    callbacks=get_callbacks(config.MODELS_DIR)
)
```

### Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted metrics
- **Confusion Matrix**: Detailed classification breakdown
- **Classification Report**: Comprehensive performance summary
- **Visualizations**: Training curves, confusion matrices, class distributions

## 🌐 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface home page |
| GET | `/health` | Health check |
| POST | `/predict` | Single image prediction |
| POST | `/predict/batch` | Batch image prediction |
| GET | `/models/info` | Model information |
| GET | `/models/classes` | Supported classes |
| POST | `/models/reload` | Reload model |
| POST | `/predict_gradcam` | Grad-CAM prediction + heatmap |
| GET | `/gradcam` | Grad-CAM web visualization |

Example Grad-CAM Response
{
    "success": true,
    "predicted_class": "healthy",
    "confidence_percentage": 95.2,
    "gradcam": {
        "main_heatmap": "<base64-image>",
        "target_layer": "mobilenetv2_1.40_224/Conv_1",
        "multi_class_heatmaps": [...]
    }
}

### Example API Usage

```python
import requests

# Single image prediction
files = {'file': open('image.jpg', 'rb')}
data = {'top_k': 3}
response = requests.post('http://localhost:5000/predict', files=files, data=data)
result = response.json()

print(f"Top prediction: {result['top_prediction']['class']}")
print(f"Confidence: {result['top_prediction']['confidence']:.4f}")
```

### cURL Examples

```bash
# Health check
curl -X GET http://localhost:5000/health

# Single image prediction
curl -X POST -F "file=@image.jpg" -F "top_k=3" http://localhost:5000/predict

# Get model information
curl -X GET http://localhost:5000/models/info
```

## 🔧 Configuration

### Key Configuration Options

```python
# config.py
class Config:
    # Image settings
    IMAGE_SIZE = (224, 224)
    CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Data augmentation
    ROTATION_RANGE = 20
    HORIZONTAL_FLIP = True
    ZOOM_RANGE = 0.2
    
    # API settings
    API_HOST = '0.0.0.0'
    API_PORT = 5000
```

## 📈 Performance Tips

### For Better Accuracy

1. **Use Transfer Learning**: EfficientNet or ResNet models typically perform better
2. **Data Augmentation**: Enable augmentation for training data
3. **Balanced Dataset**: Ensure roughly equal samples per class
4. **High-Quality Images**: Use clear, well-lit images
5. **Proper Preprocessing**: Follow the preprocessing pipeline

### For Faster Training

1. **Start with EfficientNet-B0**: Good balance of speed and accuracy
2. **Use Mixed Precision**: Enable if you have compatible GPU
3. **Optimize Batch Size**: Increase if you have sufficient GPU memory
4. **Early Stopping**: Prevent overfitting and save time

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Loading**
   ```python
   # Check if model file exists
   import os
   print(os.path.exists('models/best_model.h5'))
   ```

2. **Out of Memory Error**
   ```python
   # Reduce batch size in config.py
   BATCH_SIZE = 16  # or smaller
   ```

3. **Low Accuracy**
   - Check data quality and balance
   - Try different model architectures
   - Increase training epochs
   - Verify data preprocessing

4. **API Not Starting**
   ```bash
   # Check if port is available
   netstat -an | findstr :5000
   ```

## 📚 Advanced Usage

### Custom Model Architecture

```python
from tensorflow.keras import layers, Model

def create_custom_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, activation='relu')(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model
```

### Ensemble Predictions

```python
from src.models import create_ensemble_model

# Load multiple trained models
models = [
    keras.models.load_model('models/efficientnet_b0.h5'),
    keras.models.load_model('models/resnet_50.h5'),
    keras.models.load_model('models/densenet.h5')
]

# Create ensemble
ensemble = create_ensemble_model(models, num_classes=7)
```

### Custom Data Generator

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Custom augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)
```

## 🔒 Security Considerations

- **File Upload Validation**: Only allow image files
- **File Size Limits**: Prevent large file uploads
- **Input Sanitization**: Validate all user inputs
- **Rate Limiting**: Implement API rate limiting for production
- **Authentication**: Add authentication for production APIs

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "src/app.py"]
```

### Cloud Deployment

1. **AWS**: Use EC2 with GPU instances for training, ECS for serving
2. **Google Cloud**: Use AI Platform for training and deployment
3. **Azure**: Use Machine Learning service for end-to-end ML lifecycle

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is designed for educational and research purposes. While it can assist in preliminary disease identification, it should **never replace professional veterinary diagnosis and treatment**. Always consult qualified veterinarians for animal health issues.

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation in `/docs`

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- FastAPI team for the excellent web framework
- The open-source community for various libraries and tools
- Veterinary professionals who provided domain expertise

---

**Happy Coding! 🐾**
