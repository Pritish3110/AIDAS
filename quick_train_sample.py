"""
Quick training script for sample dataset
This will train on the small 75-image dataset for demonstration
"""
import sys
import os
sys.path.insert(0, 'src')

from config import Config
from data_preprocessing import DataPreprocessor
from models import ModelFactory
from evaluation import ModelEvaluator
import tensorflow as tf
import json

def quick_train_sample():
    """Train on sample dataset for quick demonstration"""
    
    print("🚀 Quick Training on Sample Dataset (75 images)")
    print("=" * 50)
    
    # Use sample data directory
    sample_data_dir = "data/sample"
    
    if not os.path.exists(sample_data_dir):
        print("❌ Sample dataset not found. Please run create_sample_dataset.py first")
        return
    
    # Initialize components
    config = Config()
    
    # Override config for quick training
    config.EPOCHS = 10  # Quick training
    config.BATCH_SIZE = 8  # Small batches for small dataset
    config.EARLY_STOPPING_PATIENCE = 5
    
    preprocessor = DataPreprocessor(config)
    
    print("\\n1. 📊 Analyzing sample dataset...")
    dataset_info = preprocessor.get_dataset_info(sample_data_dir)
    print(f"   Classes: {dataset_info['classes']}")
    print(f"   Total images: {dataset_info['total_images']}")
    print(f"   Distribution: {dataset_info['class_distribution']}")
    
    print("\\n2. 🔄 Organizing sample data...")
    # Organize dataset
    organized_dir = "data/sample_processed"
    class_counts = preprocessor.organize_dataset_from_folder(sample_data_dir, organized_dir)
    
    # Create splits
    preprocessor.create_train_val_test_split(organized_dir)
    
    print("\\n3. 📁 Creating datasets...")
    # Create TensorFlow datasets
    train_ds = preprocessor.create_tensorflow_dataset(config.TRAIN_DATA_DIR, augment=True)
    val_ds = preprocessor.create_tensorflow_dataset(config.VAL_DATA_DIR, augment=False)
    
    # Get number of classes
    num_classes = len(train_ds.class_names)
    print(f"   Training classes: {train_ds.class_names}")
    print(f"   Number of classes: {num_classes}")
    
    print("\\n4. 🧠 Creating model...")
    # Create a simple model for quick training
    factory = ModelFactory(config)
    model = factory.create_model('simple_cnn', num_classes)
    
    print(f"   Model parameters: {model.count_params():,}")
    
    print("\\n5. 🏋️ Training model...")
    # Train model
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS,
            verbose=1
        )
        
        print("   ✅ Training completed!")
        
        # Save model
        sample_model_path = "models/sample_model.h5"
        os.makedirs("models", exist_ok=True)
        model.save(sample_model_path)
        print(f"   Model saved: {sample_model_path}")
        
        # Save training history
        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open("models/sample_history.json", 'w') as f:
            json.dump(hist_dict, f, indent=2)
        
        print("\\n6. 📈 Training Results:")
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"   Final training accuracy: {final_acc:.4f}")
        print(f"   Final validation accuracy: {final_val_acc:.4f}")
        
        print("\\n7. 🔍 Quick Evaluation:")
        # Quick test prediction
        test_ds = preprocessor.create_tensorflow_dataset(config.TEST_DATA_DIR, augment=False)
        
        if len(list(test_ds)) > 0:
            test_loss, test_acc = model.evaluate(test_ds, verbose=0)
            print(f"   Test accuracy: {test_acc:.4f}")
        else:
            print("   No test data available (dataset too small)")
        
        print("\\n🎉 Sample training completed successfully!")
        print("\\nNext steps:")
        print("- Check models/sample_model.h5 for your trained model")
        print("- Review training history in models/sample_history.json")
        print("- Try predicting with: python -c \\"from src.inference import DiseasePredictor; p=DiseasePredictor('models/sample_model.h5'); print(p.predict_single_image('data/sample/healthy/imgs001.jpg'))\\""")
        
        return model, history
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return None, None

if __name__ == "__main__":
    quick_train_sample()