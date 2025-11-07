"""
Model architectures for animal disease classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    ResNet50, ResNet101, ResNet152,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    VGG16, VGG19, InceptionV3, DenseNet121
)
import numpy as np
from typing import Tuple, Optional, List
import logging

from config import Config

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class CustomCNN:
    """
    Custom CNN architecture for animal disease classification
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, 
                 dropout_rate: float = 0.5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def build_simple_cnn(self) -> Model:
        """
        Build a simple CNN model for baseline comparison.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            
            # First block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_advanced_cnn(self) -> Model:
        """
        Build a more advanced CNN model with residual connections.
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        # Residual blocks
        x = self._residual_block(x, 64, 3)
        x = self._residual_block(x, 128, 4, downsample=True)
        x = self._residual_block(x, 256, 6, downsample=True)
        x = self._residual_block(x, 512, 3, downsample=True)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _residual_block(self, x, filters: int, blocks: int, downsample: bool = False):
        """
        Create residual blocks for advanced CNN.
        
        Args:
            x: Input tensor
            filters: Number of filters
            blocks: Number of blocks to create
            downsample: Whether to downsample
            
        Returns:
            Output tensor
        """
        strides = 2 if downsample else 1
        
        # First block (potentially with downsampling)
        shortcut = x
        x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if downsample or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=strides)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        # Additional blocks
        for _ in range(blocks - 1):
            shortcut = x
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
        
        return x

class TransferLearningModels:
    """
    Transfer learning models using pre-trained architectures
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, 
                 dropout_rate: float = 0.5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def build_resnet_model(self, variant: str = 'ResNet50', 
                          trainable_layers: int = 20) -> Model:
        """
        Build ResNet-based transfer learning model.
        
        Args:
            variant: ResNet variant ('ResNet50', 'ResNet101', 'ResNet152')
            trainable_layers: Number of top layers to make trainable
            
        Returns:
            Compiled Keras model
        """
        # Get base model
        base_models = {
            'ResNet50': ResNet50,
            'ResNet101': ResNet101,
            'ResNet152': ResNet152
        }
        
        if variant not in base_models:
            raise ValueError(f"Unsupported ResNet variant: {variant}")
        
        base_model = base_models[variant](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        # Fine-tuning: unfreeze top layers
        if trainable_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        return model
    
    def build_efficientnet_model(self, variant: str = 'EfficientNetB0') -> Model:
        """
        Build EfficientNet-based transfer learning model.
        
        Args:
            variant: EfficientNet variant ('EfficientNetB0' to 'EfficientNetB3')
            
        Returns:
            Compiled Keras model
        """
        base_models = {
            'EfficientNetB0': EfficientNetB0,
            'EfficientNetB1': EfficientNetB1,
            'EfficientNetB2': EfficientNetB2,
            'EfficientNetB3': EfficientNetB3
        }
        
        if variant not in base_models:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")
        
        base_model = base_models[variant](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom head
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model
    
    def build_vgg_model(self, variant: str = 'VGG16') -> Model:
        """
        Build VGG-based transfer learning model.
        
        Args:
            variant: VGG variant ('VGG16' or 'VGG19')
            
        Returns:
            Compiled Keras model
        """
        base_models = {
            'VGG16': VGG16,
            'VGG19': VGG19
        }
        
        if variant not in base_models:
            raise ValueError(f"Unsupported VGG variant: {variant}")
        
        base_model = base_models[variant](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model
    
    def build_inception_model(self) -> Model:
        """
        Build InceptionV3-based transfer learning model.
        
        Returns:
            Compiled Keras model
        """
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model
    
    def build_densenet_model(self) -> Model:
        """
        Build DenseNet121-based transfer learning model.
        
        Returns:
            Compiled Keras model
        """
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class ModelFactory:
    """
    Factory class for creating different model architectures
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def create_model(self, model_type: str, num_classes: int, 
                    input_shape: Tuple[int, int, int] = None) -> Model:
        """
        Create model based on specified type.
        
        Args:
            model_type: Type of model to create
            num_classes: Number of classes
            input_shape: Input shape for the model
            
        Returns:
            Compiled Keras model
        """
        input_shape = input_shape or (*self.config.IMAGE_SIZE, self.config.CHANNELS)
        
        if model_type == 'simple_cnn':
            cnn = CustomCNN(input_shape, num_classes, self.config.DROPOUT_RATE)
            model = cnn.build_simple_cnn()
        elif model_type == 'advanced_cnn':
            cnn = CustomCNN(input_shape, num_classes, self.config.DROPOUT_RATE)
            model = cnn.build_advanced_cnn()
        elif model_type.startswith('resnet'):
            tl = TransferLearningModels(input_shape, num_classes, self.config.DROPOUT_RATE)
            variant = model_type.replace('resnet_', '').replace('_', '')
            variant = f"ResNet{variant}" if variant.isdigit() else 'ResNet50'
            model = tl.build_resnet_model(variant)
        elif model_type.startswith('efficientnet'):
            tl = TransferLearningModels(input_shape, num_classes, self.config.DROPOUT_RATE)
            variant = model_type.replace('efficientnet_', '').upper()
            variant = f"EfficientNet{variant}" if variant else 'EfficientNetB0'
            model = tl.build_efficientnet_model(variant)
        elif model_type.startswith('vgg'):
            tl = TransferLearningModels(input_shape, num_classes, self.config.DROPOUT_RATE)
            variant = model_type.upper()
            model = tl.build_vgg_model(variant)
        elif model_type == 'inception':
            tl = TransferLearningModels(input_shape, num_classes, self.config.DROPOUT_RATE)
            model = tl.build_inception_model()
        elif model_type == 'densenet':
            tl = TransferLearningModels(input_shape, num_classes, self.config.DROPOUT_RATE)
            model = tl.build_densenet_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        logger.info(f"Created {model_type} model with {model.count_params()} parameters")
        return model
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List of available model type strings
        """
        return [
            'simple_cnn',
            'advanced_cnn',
            'resnet_50',
            'resnet_101',
            'resnet_152',
            'efficientnet_b0',
            'efficientnet_b1',
            'efficientnet_b2',
            'efficientnet_b3',
            'vgg16',
            'vgg19',
            'inception',
            'densenet'
        ]

def create_ensemble_model(models: List[Model], num_classes: int) -> Model:
    """
    Create an ensemble model from multiple trained models.
    
    Args:
        models: List of trained models
        num_classes: Number of classes
        
    Returns:
        Ensemble model
    """
    # Get input shape from first model
    input_shape = models[0].input_shape[1:]
    
    # Create input layer
    inputs = layers.Input(shape=input_shape)
    
    # Get predictions from each model
    model_outputs = []
    for i, model in enumerate(models):
        # Make models non-trainable
        model.trainable = False
        output = model(inputs)
        model_outputs.append(output)
    
    # Average the predictions
    if len(model_outputs) > 1:
        averaged = layers.Average()(model_outputs)
    else:
        averaged = model_outputs[0]
    
    # Create ensemble model
    ensemble_model = Model(inputs=inputs, outputs=averaged)
    
    # Compile model
    ensemble_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return ensemble_model

if __name__ == "__main__":
    # Example usage
    config = Config()
    factory = ModelFactory(config)
    
    # Print available models
    print("Available models:")
    for model_type in factory.get_available_models():
        print(f"  - {model_type}")
    
    # Create a sample model
    try:
        model = factory.create_model('simple_cnn', num_classes=4)
        print(f"\\nCreated model with {model.count_params()} parameters")
        model.summary()
    except Exception as e:
        print(f"Error creating model: {e}")