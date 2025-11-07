# CAM Implementation Quick Start Guide

## 🎉 What's New

Your AIDAS project now includes **three** explainability visualization modes:

1. **🔍 Grad-CAM** - Gradient-based, class-specific visualizations
2. **⚡ EigenCAM** - PCA-based, fast, class-agnostic visualizations  
3. **🔬 CAM Comparison** - Side-by-side comparison of both methods

---

## 🚀 Quick Start

### 1. Start the Web Application

```bash
cd "/home/pritish_3110/Personal Files/Syllabus/Projects/AIDAS"
python enhanced_app.py
```

### 2. Access the Web Interfaces

Once the server is running, open your browser:

- **Main App**: http://localhost:5000/
- **Grad-CAM**: http://localhost:5000/gradcam
- **EigenCAM**: http://localhost:5000/eigencam
- **Comparison**: http://localhost:5000/cam_comparison

---

## 📁 Files Created

### Core Implementations
```
src/
├── gradcam.py           # Grad-CAM implementation (existing)
└── eigencam.py          # NEW: EigenCAM implementation
```

### Web Interface
```
templates/
├── gradcam_index.html      # Grad-CAM UI (updated with nav)
├── eigencam_index.html     # NEW: EigenCAM UI
└── cam_comparison.html     # NEW: Comparison UI
```

### Backend Routes (enhanced_app.py)
- `/gradcam` - Grad-CAM page
- `/predict_gradcam` - Grad-CAM API endpoint
- `/eigencam` - EigenCAM page
- `/predict_eigencam` - EigenCAM API endpoint
- `/cam_comparison` - Comparison page
- `/predict_comparison` - Comparison API endpoint

### Scripts & Documentation
```
├── compare_cam_methods.py   # NEW: CLI comparison script
├── CAM_COMPARISON.md        # NEW: Comprehensive comparison doc
└── CAM_QUICKSTART.md        # NEW: This file
```

---

## 💻 Command Line Usage

### Compare on Single Image

```bash
python compare_cam_methods.py \
    --model models/enhanced_90plus_final.h5 \
    --images data/test/sample_image.jpg \
    --output comparisons/
```

### Compare on Multiple Images

```bash
python compare_cam_methods.py \
    --model models/enhanced_90plus_final.h5 \
    --images data/test/img1.jpg data/test/img2.jpg data/test/img3.jpg \
    --output comparisons/batch_results/
```

### With Custom Alpha

```bash
python compare_cam_methods.py \
    --model models/enhanced_90plus_final.h5 \
    --images data/test/sample.jpg \
    --output comparisons/ \
    --alpha 0.5
```

---

## 🐍 Python API Usage

### Basic Usage

```python
from src.gradcam import GradCAMIntegration
from src.eigencam import EigenCAMIntegration
from tensorflow import keras
import numpy as np

# Load model
model = keras.models.load_model('models/best_model.h5')
class_names = ['healthy', 'disease1', 'disease2']

# Initialize both methods
gradcam = GradCAMIntegration(model, class_names)
eigencam = EigenCAMIntegration(model, class_names)

# Load and preprocess image
from PIL import Image
image = Image.open('test_image.jpg')
image_array = np.array(image.resize((224, 224))) / 255.0

# Generate visualizations
gradcam_result = gradcam.get_prediction_with_gradcam(image_array)
eigencam_result = eigencam.get_prediction_with_eigencam(image_array)

# Access results
print(f"Predicted: {gradcam_result['prediction']['predicted_class']}")
print(f"Confidence: {gradcam_result['prediction']['confidence']:.2%}")
```

### Save Visualizations

```python
from src.gradcam import GradCAM
from src.eigencam import EigenCAM

# Create instances
gradcam = GradCAM(model)
eigencam = EigenCAM(model)

# Generate heatmaps
gradcam_heatmap = gradcam.generate_heatmap(image_array, class_index=0)
eigencam_heatmap = eigencam.generate_heatmap(image_array, class_index=0)

# Save to files
gradcam.save_heatmap(gradcam_heatmap, 'outputs/gradcam_result.png')
eigencam.save_heatmap(eigencam_heatmap, 'outputs/eigencam_result.png')
```

---

## 🎨 Key Differences

### Grad-CAM
- ✅ **Class-specific** heatmaps
- ✅ Shows "why this class?"
- ⏱️ Slower (gradient computation)
- 🧠 Higher interpretability
- 💾 More memory usage

### EigenCAM
- ✅ **Fast** generation
- ✅ Class-agnostic approach
- ⚡ No gradient computation
- 🔍 General attention patterns
- 💾 Less memory usage

---

## 📊 Web UI Features

### All Pages Include:
- Drag-and-drop image upload
- Real-time visualization generation
- Interactive heatmap display
- Prediction confidence scores
- Multi-class visualization support

### Comparison Page Extras:
- Side-by-side heatmap comparison
- Performance metrics (speed)
- Speed advantage calculation
- Synchronized visualization display

---

## 🔧 Troubleshooting

### Issue: Import Error for EigenCAM

```bash
# Ensure you're in the project directory
cd "/home/pritish_3110/Personal Files/Syllabus/Projects/AIDAS"

# Check if eigencam.py exists
ls -l src/eigencam.py

# Test import
python -c "from src.eigencam import EigenCAM; print('Success!')"
```

### Issue: Model Not Loading

```bash
# Check available models
ls -l models/*.h5

# Update enhanced_app.py to use your model path
# Edit line 142-147 in enhanced_app.py
```

### Issue: Templates Not Found

```bash
# Verify templates exist
ls templates/eigencam_index.html
ls templates/cam_comparison.html

# Ensure enhanced_app.py is run from project root
pwd  # Should show: /home/pritish_3110/Personal Files/Syllabus/Projects/AIDAS
```

---

## 📖 Learn More

- **Comprehensive Comparison**: See `CAM_COMPARISON.md`
- **Implementation Details**: Check `src/eigencam.py` comments
- **Research Paper**: Grad-CAM - https://arxiv.org/abs/1610.02391

---

## 🎯 Typical Workflow

### For Development/Research:
1. Start with **Grad-CAM** for interpretability
2. Use **EigenCAM** for quick iterations
3. Compare both on edge cases
4. Generate batch comparisons for analysis

### For Production/Deployment:
1. Use **Grad-CAM** for clinical decisions
2. Use **EigenCAM** for screening/triage
3. Implement comparison for uncertain cases

---

## ✅ Testing Your Setup

### Quick Test Script

```python
# test_cam_implementation.py
from src.gradcam import GradCAM
from src.eigencam import EigenCAM
from tensorflow import keras
import numpy as np

# Load model
model = keras.models.load_model('models/enhanced_90plus_final.h5')
print("✓ Model loaded")

# Initialize CAMs
gradcam = GradCAM(model)
eigencam = EigenCAM(model)
print("✓ CAM methods initialized")

# Test with dummy image
dummy_image = np.random.rand(224, 224, 3)

# Test Grad-CAM
gradcam_result = gradcam.generate_heatmap(dummy_image)
print(f"✓ Grad-CAM working - Layer: {gradcam_result['target_layer']}")

# Test EigenCAM
eigencam_result = eigencam.generate_heatmap(dummy_image)
print(f"✓ EigenCAM working - Layer: {eigencam_result['target_layer']}")

print("\n🎉 All CAM implementations working correctly!")
```

Run it:
```bash
python test_cam_implementation.py
```

---

## 📞 Need Help?

- Check the detailed comparison: `CAM_COMPARISON.md`
- Review main README: `README.md`
- Examine source code: `src/gradcam.py` and `src/eigencam.py`
- Run comparison tool: `python compare_cam_methods.py --help`

---

**Happy Explaining! 🔍⚡🔬**
