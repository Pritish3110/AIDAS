"""
Animal Disease Classification System Launcher
Easy-to-use interface for all system functions
"""

import os
import sys
import argparse
import subprocess

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def print_banner():
    """Print the system banner"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║              🐾 Animal Disease Classification               ║
    ║                     System Launcher                        ║
    ╚════════════════════════════════════════════════════════════╝
    """)

def setup_data():
    """Setup sample data structure"""
    print("🔧 Setting up data structure...")
    try:
        from data_preprocessing import create_sample_dataset_structure
        create_sample_dataset_structure()
        print("✅ Sample data structure created!")
        print("📝 Please populate data/raw/ directories with your images")
    except Exception as e:
        print(f"❌ Error setting up data: {e}")

def preprocess_data():
    """Preprocess the dataset"""
    print("🔄 Preprocessing dataset...")
    try:
        from config import Config
        from data_preprocessing import DataPreprocessor
        
        config = Config()
        preprocessor = DataPreprocessor(config)
        
        # Check if raw data exists
        if not os.path.exists(config.RAW_DATA_DIR) or not os.listdir(config.RAW_DATA_DIR):
            print("❌ No raw data found. Please run 'python launch.py setup' first")
            return
        
        # Organize and split data
        class_counts = preprocessor.organize_dataset_from_folder(
            config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR
        )
        preprocessor.create_train_val_test_split(config.PROCESSED_DATA_DIR)
        
        print(f"✅ Processed {sum(class_counts.values())} images")
        print(f"📊 Class distribution: {class_counts}")
        
    except Exception as e:
        print(f"❌ Error preprocessing data: {e}")

def train_model(model_type='efficientnet_b0'):
    """Train a model"""
    print(f"🚀 Training {model_type} model...")
    try:
        from train import train
        history = train(model_type)
        print(f"✅ Training completed!")
        print(f"📈 Best validation accuracy: {max(history['val_accuracy']):.4f}")
    except Exception as e:
        print(f"❌ Training failed: {e}")

def evaluate_model():
    """Evaluate trained model"""
    print("📊 Evaluating model...")
    try:
        from config import Config
        from evaluation import evaluate_trained_model, ModelEvaluator
        
        config = Config()
        if not os.path.exists(config.BEST_MODEL_PATH):
            print("❌ No trained model found. Please train a model first.")
            return
        
        results = evaluate_trained_model(config.BEST_MODEL_PATH, config.TEST_DATA_DIR)
        
        print("✅ Evaluation completed!")
        print(f"📈 Accuracy: {results['accuracy']:.4f}")
        print(f"📈 Precision: {results['precision']:.4f}")
        print(f"📈 Recall: {results['recall']:.4f}")
        print(f"📈 F1-Score: {results['f1_score']:.4f}")
        
        # Generate report
        evaluator = ModelEvaluator(config)
        evaluator.generate_evaluation_report(results, config.MODELS_DIR)
        print(f"📄 Evaluation report saved to: {config.MODELS_DIR}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

def start_web_app():
    """Start the web application"""
    print("🌐 Starting web application...")
    app_path = os.path.join(os.path.dirname(__file__), 'src', 'app.py')
    
    if not os.path.exists(app_path):
        print("❌ Web application not found")
        return
    
    try:
        print("🚀 Web server starting at http://localhost:5000")
        print("💡 Press Ctrl+C to stop the server")
        subprocess.run([sys.executable, app_path])
    except KeyboardInterrupt:
        print("\\n🛑 Web server stopped")
    except Exception as e:
        print(f"❌ Error starting web app: {e}")

def predict_image(image_path, top_k=3):
    """Make prediction on a single image"""
    print(f"🔮 Making prediction for: {image_path}")
    try:
        from config import Config
        from inference import DiseasePredictor
        
        config = Config()
        if not os.path.exists(config.BEST_MODEL_PATH):
            print("❌ No trained model found. Please train a model first.")
            return
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        predictor = DiseasePredictor(config.BEST_MODEL_PATH, config)
        result = predictor.get_prediction_explanation(image_path)
        
        print("✅ Prediction completed!")
        print(f"🏆 Top prediction: {result['top_prediction']['class']}")
        print(f"📊 Confidence: {result['top_prediction']['confidence']:.4f}")
        print(f"💭 Explanation: {result['explanation']['explanation']}")
        print(f"💡 Recommendation: {result['explanation']['recommendation']}")
        
        # Show top predictions
        print(f"\\n📋 Top {top_k} predictions:")
        for i, pred in enumerate(result['predictions'][:top_k], 1):
            print(f"  {i}. {pred['class']}: {pred['percentage']:.1f}%")
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

def show_info():
    """Show system information"""
    print("ℹ️  System Information:")
    
    try:
        from config import Config
        config = Config()
        
        print(f"📁 Base directory: {config.BASE_DIR}")
        print(f"📊 Image size: {config.IMAGE_SIZE}")
        print(f"🎯 Supported classes: {len(config.DISEASE_CATEGORIES)}")
        
        # Check model status
        if os.path.exists(config.BEST_MODEL_PATH):
            print("✅ Trained model: Available")
        else:
            print("❌ Trained model: Not found")
        
        # Check data status
        data_dirs = ['raw', 'train', 'validation', 'test']
        for dir_name in data_dirs:
            dir_path = os.path.join(config.DATA_DIR, dir_name)
            if os.path.exists(dir_path):
                file_count = sum(len(files) for _, _, files in os.walk(dir_path))
                print(f"📂 {dir_name} data: {file_count} files")
            else:
                print(f"📂 {dir_name} data: Not found")
                
    except Exception as e:
        print(f"❌ Error getting info: {e}")

def show_models():
    """Show available model architectures"""
    print("🏗️  Available Model Architectures:")
    try:
        from models import ModelFactory
        factory = ModelFactory()
        models = factory.get_available_models()
        
        for i, model in enumerate(models, 1):
            print(f"  {i:2d}. {model}")
            
        print(f"\\n💡 Recommended: efficientnet_b0 (good balance of speed and accuracy)")
        
    except Exception as e:
        print(f"❌ Error listing models: {e}")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Animal Disease Classification System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py setup                    # Setup data structure
  python launch.py preprocess               # Preprocess dataset
  python launch.py train                    # Train with default model
  python launch.py train --model resnet_50  # Train with specific model
  python launch.py evaluate                 # Evaluate trained model
  python launch.py web                      # Start web application
  python launch.py predict image.jpg        # Predict single image
  python launch.py info                     # Show system info
  python launch.py models                   # List available models
        """
    )
    
    parser.add_argument('command', choices=[
        'setup', 'preprocess', 'train', 'evaluate', 'web', 'predict', 'info', 'models'
    ], help='Command to execute')
    
    parser.add_argument('--model', default='efficientnet_b0',
                       help='Model type for training (default: efficientnet_b0)')
    
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top predictions (default: 3)')
    
    parser.add_argument('image_path', nargs='?',
                       help='Path to image for prediction')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Execute command
    if args.command == 'setup':
        setup_data()
    elif args.command == 'preprocess':
        preprocess_data()
    elif args.command == 'train':
        train_model(args.model)
    elif args.command == 'evaluate':
        evaluate_model()
    elif args.command == 'web':
        start_web_app()
    elif args.command == 'predict':
        if not args.image_path:
            print("❌ Please provide an image path for prediction")
            parser.print_help()
        else:
            predict_image(args.image_path, args.top_k)
    elif args.command == 'info':
        show_info()
    elif args.command == 'models':
        show_models()
    
    print("\\n🐾 Thank you for using Animal Disease Classification System!")

if __name__ == "__main__":
    main()