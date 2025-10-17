"""
Enhanced Web UI for Animal Disease Classification
- Real-time image upload and prediction
- Beautiful interface with confidence scores
- Detailed disease information and recommendations
"""
import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
from PIL import Image, ImageEnhance
import json
import cv2
from werkzeug.utils import secure_filename
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src directory to path
sys.path.insert(0, 'src')

# Import Grad-CAM functionality
try:
    from gradcam import GradCAMIntegration
    GRADCAM_AVAILABLE = True
except ImportError as e:
    print(f"Grad-CAM functionality not available: {e}")
    GRADCAM_AVAILABLE = False

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'animal-disease-classifier-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
model = None
class_names = []
disease_info = {}
results_data = None
gradcam_integration = None

def load_disease_information():
    """Load disease information and recommendations"""
    return {
        'healthy': {
            'description': 'The animal appears to be in good health with no visible signs of disease.',
            'symptoms': ['Normal appearance', 'Active behavior', 'Good appetite'],
            'recommendations': [
                'Maintain regular health check-ups',
                'Ensure proper nutrition and clean water',
                'Keep vaccination schedule up to date',
                'Provide clean, comfortable living environment'
            ],
            'severity': 'None',
            'color': '#28a745'  # Green
        },
        'foot_and_mouth_disease': {
            'description': 'Foot and Mouth Disease (FMD) is a highly contagious viral disease affecting cloven-hoofed animals.',
            'symptoms': [
                'Vesicles and lesions in mouth, tongue, and lips',
                'Blisters on feet and udder',
                'High fever and loss of appetite',
                'Excessive salivation and lameness'
            ],
            'recommendations': [
                '🚨 IMMEDIATE VETERINARY ATTENTION REQUIRED',
                'Isolate affected animals immediately',
                'Report to veterinary authorities',
                'Implement strict biosecurity measures',
                'Disinfect all equipment and facilities'
            ],
            'severity': 'CRITICAL',
            'color': '#dc3545'  # Red
        },
        'lumpy_skin_disease': {
            'description': 'Lumpy Skin Disease (LSD) is a viral disease affecting cattle, characterized by skin nodules.',
            'symptoms': [
                'Firm, round nodules on skin',
                'Fever and reduced milk production',
                'Loss of appetite and weight loss',
                'Swelling of lymph nodes'
            ],
            'recommendations': [
                '⚠️ VETERINARY CONSULTATION RECOMMENDED',
                'Isolate affected animals',
                'Provide supportive care and nutrition',
                'Control insect vectors (flies, mosquitoes)',
                'Consider vaccination of healthy animals'
            ],
            'severity': 'HIGH',
            'color': '#fd7e14'  # Orange
        }
    }

def advanced_preprocess_image(image, target_size=(224, 224)):
    """Apply the same advanced preprocessing used during training"""
    
    # Convert PIL to cv2 format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = img_array
    
    # Apply CLAHE (same as training)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Noise reduction
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Convert back to PIL for enhancement
    pil_img = Image.fromarray(img)
    
    # Enhance contrast and sharpness (same as training)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.1)
    
    # Resize
    pil_img = pil_img.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    
    return img_array

def load_model():
    """Load the trained model"""
    global model, class_names, results_data, gradcam_integration
    
    # Try to load the best available model
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
                print(f"Loading model: {model_path}")
                model = tf.keras.models.load_model(model_path)
                print(f"✅ Model loaded successfully: {model_path}")
                break
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")
                continue
    
    if model is None:
        raise Exception("No trained model found! Please train a model first.")
    
    # Load class names
    results_paths = [
        'models/enhanced_90plus_results.json',
        'models/ultimate_results.json',
        'models/resnet50_results.json',
        'models/simple_custom_history.json'
    ]
    
    for results_path in results_paths:
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    class_names = results.get('class_names', ['healthy', 'foot_and_mouth_disease', 'lumpy_skin_disease'])
                    results_data = results  # Store complete results data
                print(f"✅ Class names loaded: {class_names}")
                print(f"✅ Results data loaded from: {results_path}")
                break
            except Exception as e:
                print(f"Failed to load class names from {results_path}: {e}")
                continue
    
    # Default class names if not found
    if not class_names:
        class_names = ['healthy', 'foot_and_mouth_disease', 'lumpy_skin_disease']
        print(f"⚠️ Using default class names: {class_names}")
    
    # Initialize Grad-CAM integration
    if GRADCAM_AVAILABLE and model is not None:
        try:
            gradcam_integration = GradCAMIntegration(model, class_names)
            print("✅ Grad-CAM integration initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize Grad-CAM: {e}")
            gradcam_integration = None
    else:
        gradcam_integration = None

def predict_image(image):
    """Make prediction on uploaded image"""
    try:
        # Preprocess image
        processed_img = advanced_preprocess_image(image)
        img_batch = np.expand_dims(processed_img, axis=0)
        
        # Make prediction
        predictions = model.predict(img_batch, verbose=0)
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        # Get top prediction
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Get all predictions with probabilities
        all_predictions = []
        for i, class_name in enumerate(class_names):
            all_predictions.append({
                'class': class_name,
                'probability': float(probabilities[i]),
                'percentage': float(probabilities[i] * 100)
            })
        
        # Sort by probability
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidence_percentage': confidence * 100,
            'all_predictions': all_predictions
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_plot(plot_type, data):
    """Create different types of plots based on training data"""
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
    
    if plot_type == 'training_history':
        return create_training_history_plot(data)
    elif plot_type == 'accuracy_comparison':
        return create_accuracy_comparison_plot(data)
    elif plot_type == 'loss_curves':
        return create_loss_curves_plot(data)
    elif plot_type == 'learning_rate':
        return create_learning_rate_plot(data)
    elif plot_type == 'dataset_distribution':
        return create_dataset_distribution_plot(data)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

def create_training_history_plot(data):
    """Create training history plot with accuracy and loss"""
    history = data.get('history', {})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy plot
    epochs = range(1, len(history['accuracy']) + 1)
    ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    # Convert to base64 for web display
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def create_accuracy_comparison_plot(data):
    """Create bar chart comparing final accuracies"""
    history = data.get('history', {})
    final_metrics = data.get('final_metrics', {})
    
    accuracies = {
        'Final Training': history['accuracy'][-1] if history.get('accuracy') else 0,
        'Final Validation': history['val_accuracy'][-1] if history.get('val_accuracy') else 0,
        'Best Validation': final_metrics.get('best_val_accuracy', 0),
        'Test Accuracy': final_metrics.get('test_accuracy', 0)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(accuracies.keys(), [acc * 100 for acc in accuracies.values()], 
                  color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def create_loss_curves_plot(data):
    """Create detailed loss curves plot"""
    history = data.get('history', {})
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(history['loss']) + 1)
    ax.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    
    ax.set_title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add phase separation if two phases exist
    if len(epochs) > 50:  # Assuming two-phase training
        ax.axvline(x=len(epochs)//2, color='gray', linestyle='--', alpha=0.7, label='Phase Separation')
        ax.text(len(epochs)//4, max(history['loss']) * 0.8, 'Phase 1:\nFrozen Training', 
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(len(epochs)*3//4, max(history['loss']) * 0.8, 'Phase 2:\nFine-tuning', 
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def create_learning_rate_plot(data):
    """Create learning rate schedule plot"""
    history = data.get('history', {})
    
    if 'learning_rate' not in history:
        raise ValueError("No learning rate data available")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(history['learning_rate']) + 1)
    ax.semilogy(epochs, history['learning_rate'], 'g-', linewidth=2, marker='o', markersize=3)
    
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Learning Rate (log scale)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def create_dataset_distribution_plot(data):
    """Create dataset distribution plot"""
    dataset_summary = data.get('dataset_summary', {})
    
    if not dataset_summary:
        raise ValueError("No dataset summary available")
    
    classes = list(dataset_summary.keys())
    splits = ['train', 'validation', 'test']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Stacked bar chart
    train_counts = [dataset_summary[cls]['train'] for cls in classes]
    val_counts = [dataset_summary[cls]['validation'] for cls in classes]
    test_counts = [dataset_summary[cls]['test'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.6
    
    ax1.bar(x, train_counts, width, label='Training', color='#3498db')
    ax1.bar(x, val_counts, width, bottom=train_counts, label='Validation', color='#e74c3c')
    ax1.bar(x, test_counts, width, bottom=np.array(train_counts) + np.array(val_counts), 
            label='Test', color='#2ecc71')
    
    ax1.set_title('Dataset Distribution by Class', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Number of Images')
    ax1.set_xticks(x)
    ax1.set_xticklabels([cls.replace('_', ' ').title() for cls in classes], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pie chart for total distribution
    total_counts = [dataset_summary[cls]['train'] + dataset_summary[cls]['validation'] + dataset_summary[cls]['test'] for cls in classes]
    ax2.pie(total_counts, labels=[cls.replace('_', ' ').title() for cls in classes], 
            autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c', '#2ecc71'])
    ax2.set_title('Total Dataset Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    """Main page"""
    return render_template('enhanced_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Check file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload an image.'})
        
        # Load and process image
        image = Image.open(file.stream).convert('RGB')
        
        # Make prediction
        result = predict_image(image)
        
        if result['success']:
            # Add disease information
            predicted_class = result['predicted_class']
            disease_data = disease_info.get(predicted_class, {})
            
            result.update({
                'disease_info': disease_data,
                'class_names': class_names
            })
            
            # Convert image to base64 for display
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            result['image_data'] = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing image: {str(e)}'})

@app.route('/predict_gradcam', methods=['POST'])
def predict_with_gradcam():
    """Handle image upload and prediction with Grad-CAM visualization"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Check file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload an image.'})
        
        # Check if Grad-CAM is available
        if gradcam_integration is None:
            return jsonify({
                'success': False, 
                'error': 'Grad-CAM visualization not available. Please ensure the model is loaded properly.'
            })
        
        # Load and process image
        image = Image.open(file.stream).convert('RGB')
        
        # Preprocess image for model
        processed_img = advanced_preprocess_image(image)
        
        # Get prediction with Grad-CAM
        gradcam_result = gradcam_integration.get_prediction_with_gradcam(
            processed_img, 
            top_k=3, 
            include_all_heatmaps=True
        )
        
        # Add disease information
        predicted_class = gradcam_result['prediction']['predicted_class']
        disease_data = disease_info.get(predicted_class, {})
        
        # Convert original image to base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        result = {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': gradcam_result['prediction']['confidence'],
            'confidence_percentage': gradcam_result['prediction']['confidence_percentage'],
            'all_predictions': gradcam_result['all_predictions'],
            'disease_info': disease_data,
            'class_names': class_names,
            'image_data': f"data:image/jpeg;base64,{img_base64}",
            'gradcam': {
                'main_heatmap': gradcam_result['gradcam']['main_heatmap']['superimposed_base64'],
                'heatmap_only': gradcam_result['gradcam']['main_heatmap']['heatmap_base64'],
                'target_layer': gradcam_result['gradcam']['target_layer'],
                'multi_class_heatmaps': [
                    {
                        'class_name': hm['class_name'],
                        'probability': hm['probability'],
                        'rank': hm['rank'],
                        'superimposed': hm['superimposed_base64'],
                        'heatmap_only': hm['heatmap_base64']
                    } for hm in gradcam_result['gradcam']['multi_class_heatmaps']
                ] if 'multi_class_heatmaps' in gradcam_result['gradcam'] else []
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing image with Grad-CAM: {str(e)}'})

@app.route('/gradcam')
def gradcam_page():
    """Grad-CAM visualization page"""
    return render_template('gradcam_index.html')

@app.route('/dataset')
def dataset_table():
    """Display dataset information in tabular format"""
    if results_data is None:
        return render_template('error.html', error="No training data available")
    
    dataset_summary = results_data.get('dataset_summary', {})
    config = results_data.get('config', {})
    
    # Debug: Print dataset summary to console
    print("📊 Dataset Summary Debug:")
    for class_name, data in dataset_summary.items():
        print(f"  {class_name}: train={data['train']}, val={data['validation']}, test={data['test']}")
    
    return render_template('dataset_table.html', 
                         dataset_summary=dataset_summary,
                         config=config,
                         class_names=class_names)

@app.route('/results')
def results_page():
    """Display training results and plots"""
    if results_data is None:
        return render_template('error.html', error="No training results available")
    
    return render_template('results.html', 
                         results_data=results_data,
                         class_names=class_names)

@app.route('/plot/<plot_type>')
def generate_plot(plot_type):
    """Generate and return plot images"""
    if results_data is None:
        return "No data available", 404
    
    try:
        base64_image = create_plot(plot_type, results_data)
        return base64_image
    except Exception as e:
        return f"Error generating plot: {str(e)}", 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': class_names,
        'results_loaded': results_data is not None
    })

if __name__ == '__main__':
    try:
        print("🚀 Starting Enhanced Animal Disease Classification Web App")
        print("=" * 60)
        
        # Load disease information
        disease_info = load_disease_information()
        print("✅ Disease information loaded")
        
        # Load trained model
        load_model()
        print(f"✅ Model loaded with {len(class_names)} classes")
        
        # Create templates directory if it doesn't exist
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        
        print(f"\\n🌐 Starting web server...")
        print(f"🔗 Access the application at: http://localhost:5000")
        print(f"📱 Upload animal images to get instant disease predictions")
        print(f"⚕️ Get detailed disease information and recommendations")
        print("\\n" + "=" * 60)
        
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("\\n💡 Make sure you have:")
        print("   1. Trained a model (run one of the training scripts)")
        print("   2. Model files exist in the 'models/' directory")
        print("   3. Required dependencies are installed")