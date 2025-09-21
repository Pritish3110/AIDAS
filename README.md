# Animal Disease Classification System

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

## 📁 Project Structure

```
animal-disease-classifier/
├── src/                          # Source code
│   ├── config.py                # Configuration settings
│   ├── data_preprocessing.py    # Data handling and preprocessing
│   ├── models.py               # Model architectures
│   ├── train.py                # Training pipeline
│   ├── evaluation.py           # Model evaluation and metrics
│   ├── inference.py            # Prediction and inference
│   └── app.py                  # FastAPI web application
├── data/                       # Dataset storage
│   ├── raw/                   # Raw dataset
│   ├── processed/             # Processed images
│   ├── train/                 # Training set
│   ├── validation/            # Validation set
│   └── test/                  # Test set
├── models/                     # Saved models
├── templates/                  # HTML templates
├── static/                     # Static web assets
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

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