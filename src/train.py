"""
Training pipeline for animal disease classification
"""
import os
import json
import time
import logging
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from config import Config
from data_preprocessing import DataPreprocessor
from models import ModelFactory

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def get_callbacks(models_dir: str) -> list:
    os.makedirs(models_dir, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=Config.REDUCE_LR_PATIENCE, verbose=1),
        ModelCheckpoint(filepath=Config.BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=os.path.join(models_dir, 'logs', time.strftime('%Y%m%d-%H%M%S')))
    ]
    return callbacks


def train(model_type: str = 'efficientnet_b0') -> Dict:
    """
    Train a model on the prepared dataset.

    Args:
        model_type: The type of model to train.

    Returns:
        A history dictionary containing training metrics.
    """
    preprocessor = DataPreprocessor()

    # Create datasets
    train_ds = preprocessor.create_tensorflow_dataset(Config.TRAIN_DATA_DIR, augment=True)
    val_ds = preprocessor.create_tensorflow_dataset(Config.VAL_DATA_DIR, augment=False)

    # Determine number of classes
    num_classes = len(train_ds.class_names)
    logger.info(f"Detected {num_classes} classes: {train_ds.class_names}")

    # Build model
    factory = ModelFactory()
    model = factory.create_model(model_type=model_type, num_classes=num_classes)
    model.summary(print_fn=lambda x: logger.info(x))

    # Train
    callbacks = get_callbacks(Config.MODELS_DIR)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        callbacks=callbacks
    )

    # Save final model
    final_model_path = os.path.join(Config.MODELS_DIR, f'{model_type}_final.h5')
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Save history
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = os.path.join(Config.MODELS_DIR, f'{model_type}_history.json')
    with open(history_path, 'w') as f:
        json.dump(hist, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    return hist


if __name__ == '__main__':
    # Example: train EfficientNet-B0
    train('efficientnet_b0')
