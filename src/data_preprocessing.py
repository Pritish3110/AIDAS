"""
Data preprocessing module for animal disease classification
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import shutil
from typing import Tuple, List, Dict, Optional
import logging
from tqdm import tqdm

from config import Config

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles all data preprocessing tasks including loading, augmentation,
    and dataset preparation for animal disease classification.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
    def load_and_preprocess_image(self, image_path: str, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing (width, height)
            
        Returns:
            Preprocessed image as numpy array
        """
        target_size = target_size or self.config.IMAGE_SIZE
        
        try:
            # Load image using PIL
            image = Image.open(image_path).convert('RGB')
            
            # Resize image
            image = image.resize(target_size)
            
            # Convert to numpy array
            image = np.array(image, dtype=np.float32)
            
            # Normalize pixel values to [0, 1]
            image = image / 255.0
            
            # Apply ImageNet normalization if specified
            if hasattr(self.config, 'MEAN') and hasattr(self.config, 'STD'):
                image = (image - np.array(self.config.MEAN)) / np.array(self.config.STD)
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def create_augmentation_pipeline(self) -> A.Compose:
        """
        Create an augmentation pipeline using Albumentations.
        
        Returns:
            Albumentations composition for data augmentation
        """
        transform = A.Compose([
            A.Rotate(limit=self.config.ROTATION_RANGE, p=0.7),
            A.HorizontalFlip(p=0.5 if self.config.HORIZONTAL_FLIP else 0),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.2),
            A.Normalize(mean=self.config.MEAN, std=self.config.STD, p=1.0),
        ])
        return transform
    
    def organize_dataset_from_folder(self, source_dir: str, output_dir: str = None) -> Dict[str, int]:
        """
        Organize dataset from a folder structure where subdirectories represent classes.
        
        Args:
            source_dir: Directory containing class subdirectories with images
            output_dir: Output directory for organized dataset
            
        Returns:
            Dictionary with class names and counts
        """
        output_dir = output_dir or self.config.PROCESSED_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        class_counts = {}
        
        for class_name in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_name)
            
            if not os.path.isdir(class_path):
                continue
                
            # Create class directory in output
            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            image_count = 0
            
            # Copy and potentially preprocess images
            for filename in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    src_path = os.path.join(class_path, filename)
                    dst_path = os.path.join(output_class_dir, filename)
                    
                    try:
                        # Load, preprocess and save image
                        image = self.load_and_preprocess_image(src_path)
                        if image is not None:
                            # Convert back to PIL for saving
                            if hasattr(self.config, 'MEAN') and hasattr(self.config, 'STD'):
                                # Denormalize
                                image = (image * np.array(self.config.STD)) + np.array(self.config.MEAN)
                            image = (image * 255).astype(np.uint8)
                            Image.fromarray(image).save(dst_path, quality=95)
                            image_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to process {src_path}: {str(e)}")
            
            class_counts[class_name] = image_count
            
        logger.info(f"Dataset organized. Class distribution: {class_counts}")
        return class_counts
    
    def create_train_val_test_split(self, dataset_dir: str) -> None:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset_dir: Directory containing organized dataset
        """
        # Create output directories
        for split in ['train', 'validation', 'test']:
            split_dir = os.path.join(self.config.DATA_DIR, split)
            os.makedirs(split_dir, exist_ok=True)
        
        # Process each class
        for class_name in os.listdir(dataset_dir):
            class_path = os.path.join(dataset_dir, class_name)
            
            if not os.path.isdir(class_path):
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            # Split files
            train_files, temp_files = train_test_split(
                image_files, 
                test_size=(self.config.VAL_RATIO + self.config.TEST_RATIO),
                random_state=42
            )
            
            val_files, test_files = train_test_split(
                temp_files,
                test_size=(self.config.TEST_RATIO / (self.config.VAL_RATIO + self.config.TEST_RATIO)),
                random_state=42
            )
            
            # Create class directories in each split
            for split in ['train', 'validation', 'test']:
                split_class_dir = os.path.join(self.config.DATA_DIR, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
            
            # Copy files to respective directories
            file_splits = {
                'train': train_files,
                'validation': val_files,
                'test': test_files
            }
            
            for split, files in file_splits.items():
                split_class_dir = os.path.join(self.config.DATA_DIR, split, class_name)
                
                for filename in files:
                    src_path = os.path.join(class_path, filename)
                    dst_path = os.path.join(split_class_dir, filename)
                    shutil.copy2(src_path, dst_path)
                
                logger.info(f"Copied {len(files)} files to {split}/{class_name}")
    
    def create_tensorflow_dataset(self, data_dir: str, batch_size: int = None, 
                                augment: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from directory.
        
        Args:
            data_dir: Directory containing class subdirectories
            batch_size: Batch size for dataset
            augment: Whether to apply data augmentation
            
        Returns:
            TensorFlow Dataset object
        """
        batch_size = batch_size or self.config.BATCH_SIZE
        
        # Create dataset from directory
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=None,
            seed=123,
            image_size=self.config.IMAGE_SIZE,
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        # Store class names
        self.class_names = dataset.class_names
        
        # Normalize images
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
        
        # Apply augmentation if requested
        if augment:
            augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(self.config.ROTATION_RANGE/180.0),
                tf.keras.layers.RandomZoom(self.config.ZOOM_RANGE),
            ])
            dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y))
        
        # Optimize dataset performance
        dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
    
    def get_dataset_info(self, dataset_dir: str) -> Dict:
        """
        Get information about the dataset.
        
        Args:
            dataset_dir: Directory containing the dataset
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'classes': [],
            'total_images': 0,
            'class_distribution': {}
        }
        
        for class_name in os.listdir(dataset_dir):
            class_path = os.path.join(dataset_dir, class_name)
            
            if not os.path.isdir(class_path):
                continue
            
            # Count images in class
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            count = len(image_files)
            info['classes'].append(class_name)
            info['class_distribution'][class_name] = count
            info['total_images'] += count
        
        info['num_classes'] = len(info['classes'])
        
        return info

def create_sample_dataset_structure():
    """
    Create a sample dataset structure for testing.
    This function creates empty directories that users can populate with their data.
    """
    config = Config()
    
    # Create sample class directories
    sample_classes = [
        'healthy',
        'bacterial_infection',
        'viral_infection',
        'fungal_infection'
    ]
    
    for class_name in sample_classes:
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create a README file in each directory
        readme_path = os.path.join(class_dir, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write(f"Place {class_name} images in this directory.\\n")
            f.write("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff\\n")
    
    logger.info(f"Sample dataset structure created in {config.RAW_DATA_DIR}")
    logger.info(f"Please populate the class directories with your images.")

if __name__ == "__main__":
    # Create sample dataset structure
    create_sample_dataset_structure()
    
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Get dataset info
    if os.path.exists(Config.RAW_DATA_DIR) and os.listdir(Config.RAW_DATA_DIR):
        info = preprocessor.get_dataset_info(Config.RAW_DATA_DIR)
        print("Dataset Info:", info)