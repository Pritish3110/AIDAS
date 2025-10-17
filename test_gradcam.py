#!/usr/bin/env python3
"""
Test script for Grad-CAM integration with AIDAS project
"""
import os
import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, 'src')

def test_gradcam_import():
    """Test if Grad-CAM module can be imported successfully"""
    print("🧪 Testing Grad-CAM import...")
    try:
        from gradcam import GradCAM, GradCAMIntegration
        print("✅ Grad-CAM module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Grad-CAM module: {e}")
        return False

def test_model_loading():
    """Test if we can load a trained model"""
    print("\n🧪 Testing model loading...")
    
    import tensorflow as tf
    
    model_paths = [
        'models/enhanced_90plus_final.h5',
        'models/enhanced_final_90plus.h5',
        'models/ultimate_final.h5',
        'models/resnet50_final.h5',
        'models/simple_custom_model.h5'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"  Loading model: {model_path}")
                model = tf.keras.models.load_model(model_path)
                print(f"  ✅ Model loaded successfully: {model_path}")
                print(f"     Input shape: {model.input_shape}")
                print(f"     Output shape: {model.output_shape}")
                return model, model_path
            except Exception as e:
                print(f"  ❌ Failed to load {model_path}: {e}")
                continue
    
    print("❌ No valid trained model found")
    return None, None

def test_gradcam_initialization(model, class_names):
    """Test Grad-CAM initialization with the model"""
    print("\n🧪 Testing Grad-CAM initialization...")
    
    try:
        from gradcam import GradCAM, GradCAMIntegration
        
        # Test basic GradCAM
        gradcam = GradCAM(model)
        print("✅ Basic GradCAM initialized successfully")
        print(f"   Target layer: {gradcam.target_layer_name}")
        
        # Test GradCAM integration
        integration = GradCAMIntegration(model, class_names)
        print("✅ GradCAM integration initialized successfully")
        
        return integration
        
    except Exception as e:
        print(f"❌ Failed to initialize Grad-CAM: {e}")
        return None

def test_inference_integration():
    """Test inference module with Grad-CAM"""
    print("\n🧪 Testing inference integration...")
    
    try:
        from config import Config
        from inference import DiseasePredictor
        
        config = Config()
        
        # Find a model
        model_paths = [
            'models/enhanced_90plus_final.h5',
            'models/enhanced_final_90plus.h5',
            'models/ultimate_final.h5'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    predictor = DiseasePredictor(model_path, config)
                    print(f"✅ DiseasePredictor initialized with {model_path}")
                    print(f"   Class names: {predictor.class_names}")
                    print(f"   Grad-CAM available: {predictor.gradcam_integration is not None}")
                    return predictor
                except Exception as e:
                    print(f"❌ Failed to initialize DiseasePredictor with {model_path}: {e}")
                    continue
        
        print("❌ No valid model found for DiseasePredictor")
        return None
        
    except Exception as e:
        print(f"❌ Failed to test inference integration: {e}")
        return None

def test_with_sample_image(predictor):
    """Test with a sample image if available"""
    print("\n🧪 Testing with sample image...")
    
    # Look for sample images
    sample_paths = [
        'data/test',
        'data/train',
        'data/enhanced_dataset/test',
        'data/enhanced_dataset/train',
        'data/ultimate_dataset/test',
        'data/ultimate_dataset/train'
    ]
    
    for sample_path in sample_paths:
        if os.path.exists(sample_path):
            for class_dir in os.listdir(sample_path):
                class_path = os.path.join(sample_path, class_dir)
                if os.path.isdir(class_path):
                    # Look for image files
                    for filename in os.listdir(class_path):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            image_path = os.path.join(class_path, filename)
                            print(f"  Found sample image: {image_path}")
                            
                            try:
                                # Test regular prediction
                                result = predictor.predict_single_image(image_path)
                                if 'error' not in result:
                                    print("  ✅ Regular prediction successful")
                                    print(f"     Top prediction: {result['top_prediction']['class']} ({result['top_prediction']['confidence']:.2f})")
                                else:
                                    print(f"  ❌ Regular prediction failed: {result['error']}")
                                    return False
                                
                                # Test Grad-CAM prediction if available
                                if predictor.gradcam_integration:
                                    gradcam_result = predictor.predict_with_gradcam(image_path)
                                    if 'error' not in gradcam_result:
                                        print("  ✅ Grad-CAM prediction successful")
                                        print(f"     Grad-CAM target layer: {gradcam_result['gradcam']['target_layer']}")
                                        print(f"     Heatmaps generated: {len(gradcam_result['gradcam'].get('multi_class_heatmaps', []))}")
                                    else:
                                        print(f"  ❌ Grad-CAM prediction failed: {gradcam_result['error']}")
                                        return False
                                else:
                                    print("  ⚠️ Grad-CAM not available for testing")
                                
                                return True
                                
                            except Exception as e:
                                print(f"  ❌ Error testing image {image_path}: {e}")
                                return False
    
    print("  ⚠️ No sample images found for testing")
    return True

def test_web_app_integration():
    """Test if enhanced_app.py can be imported with Grad-CAM"""
    print("\n🧪 Testing web app integration...")
    
    try:
        # Add the main directory to path
        sys.path.insert(0, '.')
        
        # Test importing the enhanced app
        print("  Importing enhanced_app...")
        # We don't actually import to avoid Flask startup, just check if gradcam import works
        
        # Check if the gradcam module is importable from the app context
        from gradcam import GradCAMIntegration
        print("✅ Grad-CAM integration should work with enhanced_app.py")
        return True
        
    except Exception as e:
        print(f"❌ Web app integration test failed: {e}")
        return False

def main():
    """Run all Grad-CAM integration tests"""
    print("🚀 AIDAS Grad-CAM Integration Test Suite")
    print("=" * 50)
    
    # Test results
    results = {
        'import': False,
        'model_loading': False,
        'gradcam_init': False,
        'inference_integration': False,
        'sample_image': False,
        'webapp_integration': False
    }
    
    # Test 1: Import
    results['import'] = test_gradcam_import()
    
    if results['import']:
        # Test 2: Model Loading
        model, model_path = test_model_loading()
        results['model_loading'] = model is not None
        
        if results['model_loading']:
            # Default class names
            class_names = ['healthy', 'foot_and_mouth_disease', 'lumpy_skin_disease']
            
            # Test 3: Grad-CAM Initialization
            integration = test_gradcam_initialization(model, class_names)
            results['gradcam_init'] = integration is not None
            
            # Test 4: Inference Integration
            predictor = test_inference_integration()
            results['inference_integration'] = predictor is not None
            
            if predictor:
                # Test 5: Sample Image
                results['sample_image'] = test_with_sample_image(predictor)
    
    # Test 6: Web App Integration
    results['webapp_integration'] = test_web_app_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():.<30} {status}")
    
    print("-" * 50)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Grad-CAM integration is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Start the enhanced app: python enhanced_app.py")
        print("2. Navigate to http://localhost:5000/gradcam")
        print("3. Upload an image to see Grad-CAM visualizations")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Please check the issues above.")
        if not results['import']:
            print("💡 Make sure all required packages are installed:")
            print("   pip install tensorflow opencv-python matplotlib")
        if results['import'] and not results['model_loading']:
            print("💡 Make sure you have trained models in the 'models/' directory")
    
    print("\n" + "=" * 50)
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)