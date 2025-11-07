"""
Complete workflow example for animal disease classification
This script demonstrates the entire process from data preparation to model deployment.
"""

import os
import sys
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from data_preprocessing import DataPreprocessor, create_sample_dataset_structure
from models import ModelFactory
from train import train
from evaluation import evaluate_trained_model, ModelEvaluator
from inference import DiseasePredictor

def main():
    """
    Complete workflow demonstration
    """
    print("🐾 Animal Disease Classification - Complete Workflow Example")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Step 1: Create sample dataset structure
    print("\\n1. Creating sample dataset structure...")
    create_sample_dataset_structure()
    print(f"   Sample dataset structure created in: {config.RAW_DATA_DIR}")
    print("   Please populate the directories with your images before proceeding.")
    
    # Check if data exists
    if not any(os.listdir(config.RAW_DATA_DIR)):
        print("   ⚠️  No data found. Please add images to the data/raw directories.")
        print("   Run this script again after adding your dataset.")
        return
    
    # Step 2: Data preprocessing
    print("\\n2. Preprocessing data...")
    preprocessor = DataPreprocessor(config)
    
    # Get dataset info
    dataset_info = preprocessor.get_dataset_info(config.RAW_DATA_DIR)
    print(f"   Dataset info: {dataset_info}")
    
    # Organize dataset
    class_counts = preprocessor.organize_dataset_from_folder(
        config.RAW_DATA_DIR, 
        config.PROCESSED_DATA_DIR
    )
    print(f"   Organized {sum(class_counts.values())} images")
    
    # Create train/val/test splits
    preprocessor.create_train_val_test_split(config.PROCESSED_DATA_DIR)
    print("   Created train/validation/test splits")
    
    # Step 3: Model creation and training
    print("\\n3. Training model...")
    
    # Show available models
    factory = ModelFactory(config)
    available_models = factory.get_available_models()
    print(f"   Available models: {available_models}")
    
    # Train a model (using EfficientNet-B0 as it's efficient and accurate)
    model_type = 'efficientnet_b0'
    print(f"   Training {model_type} model...")
    
    try:
        history = train(model_type)
        print(f"   ✅ Model trained successfully!")
        print(f"   Best validation accuracy: {max(history['val_accuracy']):.4f}")
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return
    
    # Step 4: Model evaluation
    print("\\n4. Evaluating model...")
    
    if os.path.exists(config.BEST_MODEL_PATH):
        try:
            results = evaluate_trained_model(config.BEST_MODEL_PATH, config.TEST_DATA_DIR)
            print(f"   Test accuracy: {results['accuracy']:.4f}")
            print(f"   Test precision: {results['precision']:.4f}")
            print(f"   Test recall: {results['recall']:.4f}")
            print(f"   Test F1-score: {results['f1_score']:.4f}")
            
            # Generate evaluation report
            evaluator = ModelEvaluator(config)
            evaluator.generate_evaluation_report(results, config.MODELS_DIR)
            print(f"   Evaluation report saved to: {config.MODELS_DIR}")
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
    else:
        print("   ⚠️  No trained model found for evaluation")
    
    # Step 5: Prediction examples
    print("\\n5. Making predictions...")
    
    if os.path.exists(config.BEST_MODEL_PATH):
        try:
            # Initialize predictor
            predictor = DiseasePredictor(config.BEST_MODEL_PATH, config)
            print(f"   Predictor initialized with classes: {predictor.class_names}")
            
            # Find a test image
            test_image = None
            for class_dir in os.listdir(config.TEST_DATA_DIR):
                class_path = os.path.join(config.TEST_DATA_DIR, class_dir)
                if os.path.isdir(class_path):
                    images = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                    if images:
                        test_image = os.path.join(class_path, images[0])
                        break
            
            if test_image:
                print(f"   Testing with image: {test_image}")
                
                # Single prediction
                result = predictor.get_prediction_explanation(test_image)
                
                print("   Prediction Results:")
                print(f"   - Top prediction: {result['top_prediction']['class']}")
                print(f"   - Confidence: {result['top_prediction']['confidence']:.4f}")
                print(f"   - Explanation: {result['explanation']['explanation']}")
                print(f"   - Recommendation: {result['explanation']['recommendation']}")
                
                # Save prediction results
                prediction_file = os.path.join(config.MODELS_DIR, 'sample_prediction.json')
                with open(prediction_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"   Prediction results saved to: {prediction_file}")
                
            else:
                print("   ⚠️  No test images found for prediction demo")
                
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
    else:
        print("   ⚠️  No trained model found for prediction")
    
    # Step 6: Web application setup
    print("\\n6. Web application setup...")
    app_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'app.py')
    if os.path.exists(app_file):
        print("   Web application is ready!")
        print(f"   To start the web server, run: python {app_file}")
        print("   Then visit: http://localhost:5000")
    else:
        print("   ⚠️  Web application file not found")
    
    print("\\n" + "=" * 60)
    print("🎉 Workflow completed successfully!")
    print("\\nNext steps:")
    print("1. Review the evaluation report in the models directory")
    print("2. Start the web application to test the interface")
    print("3. Use the API for integration with other systems")
    print("4. Experiment with different model architectures")
    
    return True

if __name__ == "__main__":
    main()