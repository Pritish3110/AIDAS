"""
Debug script to identify Grad-CAM initialization issues
"""
import os
import sys
import traceback

# Add src directory to path
sys.path.insert(0, 'src')

print("🔍 AIDAS Grad-CAM Debug Script")
print("=" * 50)

# Step 1: Check if gradcam module exists and can be imported
print("\n1. Testing Grad-CAM import...")
try:
    from gradcam import GradCAM, GradCAMIntegration
    print("✅ Grad-CAM module imported successfully")
    gradcam_available = True
except ImportError as e:
    print(f"❌ Failed to import Grad-CAM: {e}")
    print(f"   Error details: {traceback.format_exc()}")
    gradcam_available = False
    
# Step 2: Check TensorFlow
print("\n2. Testing TensorFlow...")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow imported successfully: {tf.__version__}")
except ImportError as e:
    print(f"❌ Failed to import TensorFlow: {e}")
    sys.exit(1)

# Step 3: Check required dependencies
print("\n3. Testing dependencies...")
dependencies = ['numpy', 'cv2', 'matplotlib', 'PIL']
missing_deps = []

for dep in dependencies:
    try:
        if dep == 'cv2':
            import cv2
            print(f"✅ OpenCV: {cv2.__version__}")
        elif dep == 'PIL':
            from PIL import Image
            print(f"✅ PIL/Pillow imported successfully")
        else:
            module = __import__(dep)
            if hasattr(module, '__version__'):
                print(f"✅ {dep}: {module.__version__}")
            else:
                print(f"✅ {dep}: imported successfully")
    except ImportError:
        missing_deps.append(dep)
        print(f"❌ Missing: {dep}")

if missing_deps:
    print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
    print("Install with: pip install " + ' '.join(missing_deps))

# Step 4: Check for trained models
print("\n4. Checking for trained models...")
model_paths = [
    'models/enhanced_90plus_final.h5',
    'models/enhanced_final_90plus.h5',
    'models/ultimate_final.h5',
    'models/resnet50_final.h5',
    'models/simple_custom_model.h5'
]

found_models = []
for model_path in model_paths:
    if os.path.exists(model_path):
        found_models.append(model_path)
        print(f"✅ Found: {model_path}")
    else:
        print(f"❌ Missing: {model_path}")

if not found_models:
    print("⚠️ No trained models found!")
    sys.exit(1)

# Step 5: Try to load a model
print("\n5. Testing model loading...")
model = None
for model_path in found_models:
    try:
        print(f"   Attempting to load: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total layers: {len(model.layers)}")
        break
    except Exception as e:
        print(f"❌ Failed to load {model_path}: {e}")
        continue

if model is None:
    print("❌ Could not load any model!")
    sys.exit(1)

# Step 6: Test Grad-CAM initialization (if available)
if gradcam_available and model is not None:
    print("\n6. Testing Grad-CAM initialization...")
    try:
        # Test basic GradCAM
        print("   Testing basic GradCAM...")
        gradcam = GradCAM(model)
        print(f"✅ Basic GradCAM initialized")
        print(f"   Target layer: {gradcam.target_layer_name}")
        
        # Test GradCAMIntegration
        print("   Testing GradCAMIntegration...")
        class_names = ['healthy', 'foot_and_mouth_disease', 'lumpy_skin_disease']
        integration = GradCAMIntegration(model, class_names)
        print(f"✅ GradCAM integration initialized successfully")
        
        print("\n🎉 Grad-CAM should work properly!")
        
    except Exception as e:
        print(f"❌ Grad-CAM initialization failed: {e}")
        print(f"   Full error: {traceback.format_exc()}")
        
        # Check model architecture for debugging
        print(f"\n📊 Model architecture analysis:")
        print(f"   Model type: {type(model)}")
        conv_layers = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'filters') or 'conv' in layer.name.lower():
                conv_layers.append((i, layer.name, type(layer).__name__))
        
        if conv_layers:
            print(f"   Found {len(conv_layers)} convolutional layers:")
            for idx, name, layer_type in conv_layers[-5:]:  # Show last 5
                print(f"     Layer {idx}: {name} ({layer_type})")
        else:
            print(f"   ⚠️ No convolutional layers found - this might be the issue!")

# Step 7: Check enhanced_app.py integration
print("\n7. Testing enhanced_app.py integration...")
try:
    # Test if the app can import gradcam
    exec("""
import os
import sys
sys.path.insert(0, 'src')
try:
    from gradcam import GradCAMIntegration
    print("✅ enhanced_app.py should be able to import GradCAMIntegration")
    integration_importable = True
except ImportError as e:
    print(f"❌ enhanced_app.py import issue: {e}")
    integration_importable = False
    """)
except Exception as e:
    print(f"❌ App integration test failed: {e}")

print("\n" + "=" * 50)
print("🏁 DIAGNOSIS COMPLETE")
print("=" * 50)

if gradcam_available and model is not None:
    print("✅ Grad-CAM should work with your setup")
    print("\n🔧 If you're still getting errors, try:")
    print("1. Restart your Flask app completely")
    print("2. Check the console output when starting enhanced_app.py")
    print("3. Look for any import errors in the startup messages")
else:
    print("❌ Issues found that need to be resolved")
    if not gradcam_available:
        print("- Fix Grad-CAM import issues")
    if model is None:
        print("- Fix model loading issues")

print(f"\n💡 Next step: Run 'python enhanced_app.py' and check startup messages")