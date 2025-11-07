"""
Configuration file for Animal Disease Classification system
"""
import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DATA_DIR = os.path.join(DATA_DIR, 'validation')
    TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)
    CHANNELS = 3
    MEAN = [0.485, 0.456, 0.406]  # ImageNet means
    STD = [0.229, 0.224, 0.225]   # ImageNet stds
    
    # Data split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    
    # Model parameters
    NUM_CLASSES = None  # Will be set based on dataset
    DROPOUT_RATE = 0.5
    
    # Data augmentation
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    HORIZONTAL_FLIP = True
    ZOOM_RANGE = 0.2
    
    # Supported disease categories (example - modify based on your dataset)
    DISEASE_CATEGORIES = [
        'bacterial_infection',
        'viral_infection',
        'fungal_infection',
        'parasitic_infection',
        'nutritional_deficiency',
        'genetic_disorder',
        'healthy'
    ]
    
    # Supported animal types
    ANIMAL_TYPES = [
        'cattle',
        'sheep',
        'goat',
        'pig',
        'poultry',
        'horse',
        'dog',
        'cat'
    ]
    
    # Model checkpoints
    BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.h5')
    CHECKPOINT_PATH = os.path.join(MODELS_DIR, 'checkpoint.h5')
    
    # API settings
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    DEBUG = True
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'