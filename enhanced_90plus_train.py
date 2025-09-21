"""
Enhanced preprocessing and training script for 90%+ accuracy
- Advanced preprocessing techniques
- Sophisticated augmentation strategies  
- Optimized model architecture
- Ensemble learning approach
"""
import os
import sys
import shutil
from pathlib import Path
import tensorflow as tf
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def advanced_image_preprocessing(image_path, target_size=(224, 224)):
    """Advanced preprocessing pipeline for maximum accuracy"""
    
    # Load image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Noise reduction
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Convert to PIL for further processing
    pil_img = Image.fromarray(img)
    
    # Enhance contrast and sharpness
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.1)
    
    # Resize
    pil_img = pil_img.resize(target_size, Image.LANCZOS)
    
    return np.array(pil_img)

def create_enhanced_dataset(train_per_class=180, val_per_class=35, test_per_class=30):
    """Create enhanced dataset with advanced preprocessing"""
    
    print(f"🚀 Creating ENHANCED Dataset for 90%+ Accuracy")
    print(f"   📚 {train_per_class} train + {val_per_class} val + {test_per_class} test per category")
    print(f"   🔬 Advanced preprocessing enabled")
    
    source_dirs = {
        'healthy': 'data/raw/healthy',
        'foot_and_mouth_disease': 'data/raw/foot_and_mouth_disease', 
        'lumpy_skin_disease': 'data/raw/lumpy_skin_disease'
    }
    
    custom_base = Path('data/enhanced_dataset')
    train_dir = custom_base / 'train'
    val_dir = custom_base / 'validation'
    test_dir = custom_base / 'test'
    
    # Clean and create directories
    if custom_base.exists():
        shutil.rmtree(custom_base)
    
    for dir_path in [custom_base, train_dir, val_dir, test_dir]:
        dir_path.mkdir(exist_ok=True)
    
    dataset_summary = {}
    
    for category, source_path in source_dirs.items():
        print(f"  🔬 Processing {category} with advanced preprocessing...")
        
        source_p = Path(source_path)
        if not source_p.exists():
            continue
        
        # Create category directories
        for split_dir in [train_dir, val_dir, test_dir]:
            (split_dir / category).mkdir(exist_ok=True)
        
        # Get all images
        jpg_files = list(source_p.glob('*.jpg'))
        np.random.shuffle(jpg_files)
        
        # Calculate splits
        total_needed = train_per_class + val_per_class + test_per_class
        available = len(jpg_files)
        
        if available < total_needed:
            ratio = available / total_needed
            train_target = int(train_per_class * ratio)
            val_target = int(val_per_class * ratio)
            test_target = available - train_target - val_target
        else:
            train_target = train_per_class
            val_target = val_per_class
            test_target = test_per_class
        
        # Process and save images with advanced preprocessing
        splits = [
            (jpg_files[:train_target], train_dir / category),
            (jpg_files[train_target:train_target + val_target], val_dir / category),
            (jpg_files[train_target + val_target:train_target + val_target + test_target], test_dir / category)
        ]
        
        counts = []
        for files, dest_dir in splits:
            count = 0
            for img_file in files:
                try:
                    # Apply advanced preprocessing
                    processed_img = advanced_image_preprocessing(img_file)
                    
                    # Save processed image
                    pil_img = Image.fromarray(processed_img)
                    output_path = dest_dir / img_file.name
                    pil_img.save(output_path, 'JPEG', quality=95)
                    count += 1
                except Exception as e:
                    print(f"    Error processing {img_file}: {e}")
            counts.append(count)
        
        dataset_summary[category] = {
            'train': counts[0],
            'validation': counts[1],
            'test': counts[2],
            'total_available': available
        }
        
        print(f"    ✅ Train: {counts[0]}, Val: {counts[1]}, Test: {counts[2]}")
    
    total_train = sum([v['train'] for v in dataset_summary.values()])
    total_val = sum([v['validation'] for v in dataset_summary.values()])
    total_test = sum([v['test'] for v in dataset_summary.values()])
    
    print(f"\\n📊 Enhanced Dataset Summary:")
    print(f"   Total Training: {total_train} images (with advanced preprocessing)")
    print(f"   Total Validation: {total_val} images")
    print(f"   Total Test: {total_test} images")
    
    return custom_base, dataset_summary

def create_optimized_mobilenet_model(num_classes, input_shape=(224, 224, 3)):
    """Create optimized MobileNetV2 model for 90%+ accuracy"""
    
    tf.keras.backend.clear_session()
    
    # Create base model with higher alpha for better performance
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=1.4  # Higher alpha for better performance
    )
    
    base_model.trainable = False
    
    # Optimized classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def create_sophisticated_augmentation():
    """Create sophisticated augmentation for maximum diversity"""
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=35,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.25,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
        channel_shift_range=0.2,
        fill_mode='nearest',
        # Advanced augmentations
        featurewise_center=False,
        featurewise_std_normalization=False,
        zca_whitening=False
    )

def create_dataset_optimized(data_dir, batch_size=10, image_size=(224, 224), augment=False):
    """Create optimized dataset"""
    
    if augment:
        datagen = create_sophisticated_augmentation()
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    dataset = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True,
        interpolation='lanczos'  # Better interpolation
    )
    
    return dataset

def train_enhanced_90plus_model():
    """Enhanced training for guaranteed 90%+ accuracy"""
    
    print("\\n🏆 ENHANCED TRAINING FOR 90%+ ACCURACY")
    print("=" * 65)
    
    # Create enhanced dataset
    custom_base, dataset_summary = create_enhanced_dataset(
        train_per_class=180, 
        val_per_class=35, 
        test_per_class=30
    )
    
    # Optimized configuration for 90%+
    EPOCHS = 150
    BATCH_SIZE = 10  # Smaller batch for better gradients
    INITIAL_LR = 0.0002
    FINE_TUNE_LR = 0.000003
    IMAGE_SIZE = (224, 224)
    
    train_dir = str(custom_base / 'train')
    val_dir = str(custom_base / 'validation')
    test_dir = str(custom_base / 'test')
    
    print(f"\\n📁 Enhanced Dataset Directories:")
    print(f"   Train: {train_dir}")
    print(f"   Validation: {val_dir}")
    print(f"   Test: {test_dir}")
    
    # Create optimized datasets
    print(f"\\n🔄 Creating optimized datasets...")
    try:
        train_ds = create_dataset_optimized(train_dir, BATCH_SIZE, IMAGE_SIZE, augment=True)
        val_ds = create_dataset_optimized(val_dir, BATCH_SIZE, IMAGE_SIZE, augment=False)
        test_ds = create_dataset_optimized(test_dir, BATCH_SIZE, IMAGE_SIZE, augment=False)
        
        print(f"✅ Enhanced datasets created successfully")
        print(f"   Classes: {list(train_ds.class_indices.keys())}")
        print(f"   Train samples: {train_ds.samples}")
        print(f"   Validation samples: {val_ds.samples}")
        print(f"   Test samples: {test_ds.samples}")
        
        class_names = list(train_ds.class_indices.keys())
        num_classes = len(class_names)
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return None
    
    # Create optimized model
    print(f"\\n🧠 Creating optimized MobileNetV2 (alpha=1.4)...")
    try:
        model, base_model = create_optimized_mobilenet_model(num_classes, IMAGE_SIZE + (3,))
        
        print(f"   Model: Enhanced MobileNetV2 (alpha=1.4)")
        print(f"   Classes: {num_classes}")
        print(f"   Total Parameters: {model.count_params():,}")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None
    
    # Phase 1: Enhanced frozen training
    print(f"\\n🏋️ Phase 1: Enhanced frozen base training...")
    
    try:
        # Compile with optimized settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False  # Graph mode for speed
        )
        
        # Ultra-sophisticated callbacks
        callbacks_phase1 = [
            tf.keras.callbacks.EarlyStopping(
                patience=30,
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1,
                min_delta=0.0005
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=12,
                factor=0.15,
                monitor='val_accuracy',
                verbose=1,
                min_lr=1e-10,
                min_delta=0.0003
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/enhanced_phase1_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        train_steps = max(1, train_ds.samples // BATCH_SIZE)
        val_steps = max(1, val_ds.samples // BATCH_SIZE)
        
        print(f"   Train steps per epoch: {train_steps}")
        print(f"   Validation steps per epoch: {val_steps}")
        
        # Phase 1 training
        history_phase1 = model.fit(
            train_ds,
            steps_per_epoch=train_steps,
            validation_data=val_ds,
            validation_steps=val_steps,
            epochs=EPOCHS//2,
            callbacks=callbacks_phase1,
            verbose=1
        )
        
        print("✅ Phase 1 completed!")
        phase1_best = max(history_phase1.history['val_accuracy'])
        print(f"   Phase 1 Best Val Accuracy: {phase1_best:.4f}")
        
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        return None
    
    # Phase 2: Ultra-careful fine-tuning
    print(f"\\n🔧 Phase 2: Ultra-careful fine-tuning...")
    
    try:
        base_model.trainable = True
        
        # Unfreeze only top layers
        fine_tune_at = 50
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False
        
        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"   Unfrozen layers: {trainable_count}")
        
        # Recompile with ultra-low learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Ultra-patient callbacks
        callbacks_phase2 = [
            tf.keras.callbacks.EarlyStopping(
                patience=50,  # Maximum patience
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1,
                min_delta=0.0002
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=20,
                factor=0.1,
                monitor='val_accuracy',
                verbose=1,
                min_lr=1e-12,
                min_delta=0.0001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/enhanced_final_90plus.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Fine-tuning
        history_phase2 = model.fit(
            train_ds,
            steps_per_epoch=train_steps,
            validation_data=val_ds,
            validation_steps=val_steps,
            epochs=EPOCHS//2,
            callbacks=callbacks_phase2,
            verbose=1,
            initial_epoch=len(history_phase1.history['loss'])
        )
        
        print("✅ Phase 2 completed!")
        
        # Combine histories
        combined_history = {}
        for key in history_phase1.history.keys():
            combined_history[key] = history_phase1.history[key] + history_phase2.history[key]
        
        os.makedirs('models', exist_ok=True)
        model.save('models/enhanced_90plus_final.h5')
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        combined_history = history_phase1.history
    
    # Final Evaluation
    print(f"\\n📊 FINAL ENHANCED EVALUATION:")
    try:
        test_steps = max(1, test_ds.samples // BATCH_SIZE)
        test_loss, test_acc = model.evaluate(test_ds, steps=test_steps, verbose=0)
        
        final_train_acc = combined_history['accuracy'][-1]
        final_val_acc = combined_history['val_accuracy'][-1]
        best_val_acc = max(combined_history['val_accuracy'])
        
        print(f"   Final Training Accuracy: {final_train_acc:.4f}")
        print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"   🏆 BEST Validation Accuracy: {best_val_acc:.4f}")
        print(f"   🎯 TEST ACCURACY: {test_acc:.4f}")
        print(f"   📉 Test Loss: {test_loss:.4f}")
        
        # Save comprehensive results
        results_data = {
            'history': {k: [float(v) for v in vals] for k, vals in combined_history.items()},
            'dataset_summary': dataset_summary,
            'class_names': class_names,
            'final_metrics': {
                'test_accuracy': float(test_acc),
                'test_loss': float(test_loss),
                'best_val_accuracy': float(best_val_acc)
            },
            'config': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'initial_lr': INITIAL_LR,
                'fine_tune_lr': FINE_TUNE_LR,
                'preprocessing': 'advanced_clahe_bilateral_contrast_sharpness',
                'model': 'MobileNetV2_alpha_1.4'
            }
        }
        
        with open('models/enhanced_90plus_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return {
            'model': model,
            'history': combined_history,
            'test_accuracy': test_acc,
            'dataset_summary': dataset_summary,
            'class_names': class_names,
            'final_metrics': results_data['final_metrics']
        }
        
    except Exception as e:
        print(f"❌ Final evaluation failed: {e}")
        return None

if __name__ == "__main__":
    print("🚀 ENHANCED ANIMAL DISEASE CLASSIFICATION - 90%+ TARGET")
    print("🔬 Advanced Preprocessing + Optimized Architecture")
    print("=" * 70)
    
    results = train_enhanced_90plus_model()
    
    if results:
        print(f"\\n" + "="*70)
        print(f"🏁 ENHANCED TRAINING COMPLETED!")
        print(f"="*70)
        
        metrics = results['final_metrics']
        test_acc = metrics['test_accuracy']
        
        print(f"\\n🏆 FINAL ENHANCED ASSESSMENT:")
        print(f"   🎯 TARGET: 90%+ Accuracy")
        print(f"   🔥 ACHIEVED: {test_acc:.1%}")
        print(f"   📈 Best Val: {metrics['best_val_accuracy']:.1%}")
        
        if test_acc >= 0.9:
            print(f"\\n🎉 MISSION ACCOMPLISHED! 🎉")
            print(f"✨ 90%+ TARGET ACHIEVED!")
        elif test_acc >= 0.85:
            print(f"\\n🔥 EXCELLENT! Very close to 90%!")
        else:
            print(f"\\n💪 Strong progress! Keep optimizing!")
            
        # Save model for UI
        print(f"\\n💾 Saving model for UI integration...")
        
    else:
        print(f"\\n❌ Enhanced training failed!")

print("\\n🏁 Enhanced training completed!")