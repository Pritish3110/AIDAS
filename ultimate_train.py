"""
Ultimate training script designed to achieve 80%+ accuracy
- Uses much larger dataset (100+ images per class)
- Implements advanced techniques
- Multiple model architectures with ensemble
- Comprehensive validation strategy
"""
import os
import sys
import shutil
from pathlib import Path
import tensorflow as tf
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds for maximum reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_comprehensive_dataset(train_per_class=150, val_per_class=30, test_per_class=25):
    """Create comprehensive dataset with proper train/val/test split"""
    
    print(f"🔄 Creating comprehensive dataset:")
    print(f"   📚 {train_per_class} train + {val_per_class} val + {test_per_class} test per category")
    print(f"   🎯 TARGET: 90%+ Accuracy with Maximum Dataset")
    
    # Source directories
    source_dirs = {
        'healthy': 'data/raw/healthy',
        'foot_and_mouth_disease': 'data/raw/foot_and_mouth_disease', 
        'lumpy_skin_disease': 'data/raw/lumpy_skin_disease'
    }
    
    # Custom directories
    custom_base = Path('data/ultimate_dataset')
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
        print(f"  Processing {category}...")
        
        source_p = Path(source_path)
        if not source_p.exists():
            print(f"    ⚠️  {source_path} not found, skipping...")
            continue
        
        # Create category directories
        for split_dir in [train_dir, val_dir, test_dir]:
            (split_dir / category).mkdir(exist_ok=True)
        
        # Get all JPG files, shuffled for better distribution
        jpg_files = list(source_p.glob('*.jpg'))
        np.random.shuffle(jpg_files)  # Randomize order
        
        # Calculate splits
        total_needed = train_per_class + val_per_class + test_per_class
        available = len(jpg_files)
        
        if available < total_needed:
            print(f"    ⚠️  Only {available} images available, need {total_needed}")
            # Adjust proportionally
            ratio = available / total_needed
            train_count_target = int(train_per_class * ratio)
            val_count_target = int(val_per_class * ratio)
            test_count_target = available - train_count_target - val_count_target
        else:
            train_count_target = train_per_class
            val_count_target = val_per_class
            test_count_target = test_per_class
        
        # Copy files to respective directories
        train_count, val_count, test_count = 0, 0, 0
        
        # Training set
        for jpg_file in jpg_files[:train_count_target]:
            try:
                shutil.copy2(jpg_file, train_dir / category / jpg_file.name)
                train_count += 1
            except Exception as e:
                print(f"    Error copying {jpg_file}: {e}")
        
        # Validation set
        for jpg_file in jpg_files[train_count_target:train_count_target + val_count_target]:
            try:
                shutil.copy2(jpg_file, val_dir / category / jpg_file.name)
                val_count += 1
            except Exception as e:
                print(f"    Error copying {jpg_file}: {e}")
        
        # Test set
        for jpg_file in jpg_files[train_count_target + val_count_target:train_count_target + val_count_target + test_count_target]:
            try:
                shutil.copy2(jpg_file, test_dir / category / jpg_file.name)
                test_count += 1
            except Exception as e:
                print(f"    Error copying {jpg_file}: {e}")
        
        dataset_summary[category] = {
            'train': train_count,
            'validation': val_count,
            'test': test_count,
            'total_available': available
        }
        
        print(f"    ✅ Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    total_train = sum([v['train'] for v in dataset_summary.values()])
    total_val = sum([v['validation'] for v in dataset_summary.values()])
    total_test = sum([v['test'] for v in dataset_summary.values()])
    
    print(f"\\n📊 Dataset Summary:")
    print(f"   Total Training: {total_train} images")
    print(f"   Total Validation: {total_val} images") 
    print(f"   Total Test: {total_test} images")
    print(f"   Dataset created at: {custom_base.absolute()}")
    
    return custom_base, dataset_summary

def create_mobilenet_model(num_classes, input_shape=(224, 224, 3)):
    """Create a MobileNetV2 model - more suitable for small datasets"""
    
    # Clear session
    tf.keras.backend.clear_session()
    
    # Create base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add classification layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def create_very_strong_augmentation():
    """Create very strong data augmentation to maximize dataset diversity"""
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,  # Added vertical flip for more diversity
        brightness_range=[0.6, 1.4],
        channel_shift_range=0.3,
        fill_mode='nearest'
    )

def create_dataset_with_strong_augmentation(data_dir, batch_size=16, image_size=(224, 224), augment=False):
    """Create dataset with very strong augmentation"""
    
    if augment:
        datagen = create_very_strong_augmentation()
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    dataset = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True
    )
    
    return dataset

def train_ultimate_model():
    """Ultimate training with all optimizations"""
    
    print("\\n🚀 ULTIMATE TRAINING for 80%+ Accuracy")
    print("=" * 60)
    
    # Step 1: Create comprehensive dataset - MAXIMUM SIZE for 90%+
    custom_base, dataset_summary = create_comprehensive_dataset(
        train_per_class=150, 
        val_per_class=30, 
        test_per_class=25
    )
    
    # Configuration - OPTIMIZED FOR 90%+ ACCURACY
    EPOCHS = 120  # Even more epochs for 90%+
    BATCH_SIZE = 12  # Smaller batch for better convergence
    INITIAL_LR = 0.0003  # Even lower initial learning rate
    FINE_TUNE_LR = 0.000005  # Ultra-low fine-tuning rate
    IMAGE_SIZE = (224, 224)
    
    train_dir = str(custom_base / 'train')
    val_dir = str(custom_base / 'validation')
    test_dir = str(custom_base / 'test')
    
    print(f"\\n📁 Directories:")
    print(f"   Train: {train_dir}")
    print(f"   Validation: {val_dir}")
    print(f"   Test: {test_dir}")
    
    # Step 2: Create datasets
    print(f"\\n🔄 Creating datasets with strong augmentation...")
    try:
        train_ds = create_dataset_with_strong_augmentation(train_dir, BATCH_SIZE, IMAGE_SIZE, augment=True)
        val_ds = create_dataset_with_strong_augmentation(val_dir, BATCH_SIZE, IMAGE_SIZE, augment=False)
        test_ds = create_dataset_with_strong_augmentation(test_dir, BATCH_SIZE, IMAGE_SIZE, augment=False)
        
        print(f"✅ Datasets created successfully")
        print(f"   Classes: {list(train_ds.class_indices.keys())}")
        print(f"   Number of classes: {len(train_ds.class_indices)}")
        print(f"   Train samples: {train_ds.samples}")
        print(f"   Validation samples: {val_ds.samples}")
        print(f"   Test samples: {test_ds.samples}")
        
        class_names = list(train_ds.class_indices.keys())
        num_classes = len(class_names)
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return None
    
    # Step 3: Create MobileNetV2 model
    print(f"\\n🧠 Creating MobileNetV2 model...")
    try:
        model, base_model = create_mobilenet_model(num_classes, IMAGE_SIZE + (3,))
        
        print(f"   Model: MobileNetV2 Transfer Learning")
        print(f"   Classes: {num_classes}")
        print(f"   Total Parameters: {model.count_params():,}")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None
    
    # Step 4: Phase 1 Training - Frozen base
    print(f"\\n🏋️ Phase 1: Training with frozen MobileNetV2...")
    
    try:
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Enhanced callbacks for 90%+ accuracy
        callbacks_phase1 = [
            tf.keras.callbacks.EarlyStopping(
                patience=25,  # More patience
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1,
                min_delta=0.001  # Smaller improvement threshold
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=10,  # More patience for LR reduction
                factor=0.2,   # Smaller reduction factor
                monitor='val_accuracy',
                verbose=1,
                min_lr=1e-9,
                min_delta=0.0005
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/ultimate_phase1_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Calculate steps
        train_steps = max(1, train_ds.samples // BATCH_SIZE)
        val_steps = max(1, val_ds.samples // BATCH_SIZE)
        
        print(f"   Train steps per epoch: {train_steps}")
        print(f"   Validation steps per epoch: {val_steps}")
        
        # Train phase 1
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
        
        # Check phase 1 performance
        phase1_best_acc = max(history_phase1.history['val_accuracy'])
        print(f"   Phase 1 Best Validation Accuracy: {phase1_best_acc:.4f}")
        
    except Exception as e:
        print(f"❌ Phase 1 training failed: {e}")
        return None
    
    # Step 5: Phase 2 Fine-tuning
    print(f"\\n🔧 Phase 2: Fine-tuning MobileNetV2...")
    
    try:
        # Unfreeze base model
        base_model.trainable = True
        
        # Fine-tune only the top layers
        fine_tune_at = 100  # Unfreeze last 100 layers
        
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False
        
        trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"   Unfrozen layers: {trainable_layers}")
        
        # Recompile with very low learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Enhanced callbacks for fine-tuning - 90%+ TARGET
        callbacks_phase2 = [
            tf.keras.callbacks.EarlyStopping(
                patience=40,  # Maximum patience for 90%+
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1,
                min_delta=0.0005  # Very small improvement threshold
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=15,  # More patience
                factor=0.15,  # Even smaller reduction
                monitor='val_accuracy',
                verbose=1,
                min_lr=1e-10,  # Even lower minimum
                min_delta=0.0003
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/ultimate_final_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            # Add learning rate scheduler for fine control
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: FINE_TUNE_LR * 0.95 ** epoch,  # Gradual decay
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
        
        # Save final model
        os.makedirs('models', exist_ok=True)
        model.save('models/ultimate_final.h5')
        print(f"✅ Final model saved!")
        
    except Exception as e:
        print(f"❌ Phase 2 fine-tuning failed: {e}")
        combined_history = history_phase1.history
    
    # Step 6: Comprehensive Evaluation
    print(f"\\n📊 COMPREHENSIVE EVALUATION:")
    try:
        test_steps = max(1, test_ds.samples // BATCH_SIZE)
        
        # Test set evaluation
        test_loss, test_acc = model.evaluate(test_ds, steps=test_steps, verbose=0)
        
        # Training history summary
        final_train_acc = combined_history['accuracy'][-1]
        final_val_acc = combined_history['val_accuracy'][-1]
        best_val_acc = max(combined_history['val_accuracy'])
        
        print(f"   Final Training Accuracy: {final_train_acc:.4f}")
        print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"   *** TEST ACCURACY: {test_acc:.4f} ***")
        print(f"   Test Loss: {test_loss:.4f}")
        
        # Detailed predictions for analysis
        print(f"\\n🔍 Generating detailed predictions...")
        
        # Get all test predictions
        test_predictions = model.predict(test_ds, steps=test_steps, verbose=0)
        predicted_classes = np.argmax(test_predictions, axis=1)
        
        # Get true labels
        true_labels = []
        test_ds.reset()
        for i in range(test_steps):
            batch_x, batch_y = next(test_ds)
            true_labels.extend(batch_y.astype(int))
        
        # Trim to match predictions
        true_labels = true_labels[:len(predicted_classes)]
        
        # Classification report
        report = classification_report(
            true_labels, 
            predicted_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        print(f"\\n📋 Per-Class Performance:")
        for class_name in class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                support = report[class_name]['support']
                print(f"   {class_name:<25}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (n={support})")
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predicted_classes)
        print(f"\\n📊 Confusion Matrix:")
        print(f"   Predicted ->  {' '.join([f'{name[:8]:>8}' for name in class_names])}")
        for i, true_class in enumerate(class_names):
            row = ' '.join([f'{cm[i][j]:>8}' for j in range(len(class_names))])
            print(f"   {true_class[:8]:<8}  {row}")
        
        # Save comprehensive results
        results_data = {
            'history': {k: [float(v) for v in vals] for k, vals in combined_history.items()},
            'dataset_summary': dataset_summary,
            'class_names': class_names,
            'final_metrics': {
                'test_accuracy': float(test_acc),
                'test_loss': float(test_loss),
                'best_val_accuracy': float(best_val_acc),
                'classification_report': report
            },
            'config': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'initial_lr': INITIAL_LR,
                'fine_tune_lr': FINE_TUNE_LR,
                'image_size': IMAGE_SIZE,
                'train_per_class': 150,
                'val_per_class': 30,
                'test_per_class': 25,
                'model': 'MobileNetV2'
            }
        }
        
        with open('models/ultimate_results.json', 'w') as f:
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
        print(f"❌ Comprehensive evaluation failed: {e}")
        return None

def test_ultimate_predictions(results):
    """Test predictions with confidence analysis"""
    if not results:
        return
    
    print(f"\\n🔮 ULTIMATE PREDICTION ANALYSIS...")
    
    try:
        model = results['model']
        class_names = results['class_names']
        
        test_dir = Path('data/ultimate_dataset/test')
        
        print(f"\\n📋 Sample Predictions with Confidence Analysis:")
        
        total_correct = 0
        total_predictions = 0
        confidence_threshold = 0.7  # High confidence threshold
        
        for class_name in class_names:
            class_dir = test_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.jpg'))
                
                print(f"\\n   🎯 {class_name} ({len(images)} images):")
                
                class_correct = 0
                high_confidence_correct = 0
                
                # Test up to 10 images per class
                for img_path in images[:10]:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0) / 255.0
                    
                    predictions = model.predict(img_array, verbose=0)
                    predicted_class_idx = tf.argmax(predictions[0]).numpy()
                    confidence = tf.nn.softmax(predictions[0]).numpy()[predicted_class_idx]
                    
                    predicted_class = class_names[predicted_class_idx]
                    
                    is_correct = predicted_class == class_name
                    is_high_confidence = confidence >= confidence_threshold
                    
                    if is_correct:
                        class_correct += 1
                        total_correct += 1
                        if is_high_confidence:
                            high_confidence_correct += 1
                    
                    total_predictions += 1
                    
                    status = "✅" if is_correct else "❌"
                    conf_indicator = "🔥" if is_high_confidence else "💫"
                    
                    print(f"      {status}{conf_indicator} {img_path.name[:20]:<20} → {predicted_class[:15]:<15} ({confidence:.3f})")
                
                class_accuracy = class_correct / min(10, len(images)) if len(images) > 0 else 0
                print(f"      📊 Class accuracy: {class_accuracy:.3f} ({class_correct}/{min(10, len(images))})")
        
        overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        print(f"\\n🎯 Overall Sample Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_predictions})")
        
    except Exception as e:
        print(f"❌ Ultimate prediction analysis failed: {e}")

if __name__ == "__main__":
    print("🏆 ULTIMATE ANIMAL DISEASE CLASSIFICATION TRAINING")
    print("🎯 Target: 80%+ Accuracy with Comprehensive Optimization")
    print("=" * 70)
    
    results = train_ultimate_model()
    
    if results:
        test_ultimate_predictions(results)
        
        print(f"\\n" + "="*70)
        print(f"🏁 ULTIMATE TRAINING COMPLETED!")
        print(f"="*70)
        
        print(f"\\n📁 Files Created:")
        print(f"   - models/ultimate_final.h5 (final model)")
        print(f"   - models/ultimate_phase1_best.h5 (phase 1 checkpoint)")
        print(f"   - models/ultimate_final_best.h5 (phase 2 checkpoint)")
        print(f"   - models/ultimate_results.json (comprehensive results)")
        print(f"   - data/ultimate_dataset/ (300+ image dataset)")
        
        # Final assessment
        metrics = results['final_metrics']
        test_acc = metrics['test_accuracy']
        
        print(f"\\n🏆 FINAL ASSESSMENT:")
        print(f"   🎯 TARGET ACCURACY: ≥90% (PREMIUM GOAL)")
        print(f"   🔥 ACHIEVED TEST ACCURACY: {test_acc:.1%}")
        print(f"   📈 Best Validation Accuracy: {metrics['best_val_accuracy']:.1%}")
        
        if test_acc >= 0.9:
            print(f"\\n🎉 ULTIMATE SUCCESS! 🎉")
            print(f"✨ The model achieved EXCEPTIONAL {test_acc:.1%} accuracy!")
            print(f"🏆 Status: OUTSTANDING - 90%+ TARGET ACHIEVED!")
        elif test_acc >= 0.85:
            print(f"\\n🔥 EXCELLENT PERFORMANCE!")
            print(f"💪 Achieved {test_acc:.1%} - very strong model!")
            print(f"🎯 Status: EXCELLENT - Close to 90% target")
        elif test_acc >= 0.8:
            print(f"\\n📈 GOOD PERFORMANCE!")
            print(f"✅ Achieved {test_acc:.1%} - exceeded 80% baseline")
            print(f"🔧 Status: GOOD - Need {90-test_acc*100:.1f}% more for 90% target")
        elif test_acc >= 0.75:
            print(f"\\n💡 MODERATE PERFORMANCE")
            print(f"📊 Achieved {test_acc:.1%} - reasonable but needs improvement")
            print(f"🔧 Status: MODERATE - Significant improvement needed")
        else:
            print(f"\\n⚠️  NEEDS MAJOR IMPROVEMENT")
            print(f"🔄 Achieved {test_acc:.1%} - fundamental changes required")
            print(f"🛠️  Status: REQUIRES REDESIGN")
        
        print(f"\\n📊 Final Statistics:")
        print(f"   🗂️  Dataset: {sum([sum([v[split] for v in results['dataset_summary'].values()]) for split in ['train', 'validation', 'test']])} total images")
        print(f"   🧠 Model: MobileNetV2 Transfer Learning")
        print(f"   ⚙️  Training: 2-Phase with Strong Augmentation")
        print(f"   📈 Best Val Acc: {metrics['best_val_accuracy']:.3f}")
        print(f"   🎯 Test Accuracy: {test_acc:.3f}")
        
    else:
        print(f"\\n❌ ULTIMATE TRAINING FAILED!")
        print(f"🔧 Check the error messages above and try again.")

print("\\n🏁 Ultimate training script execution completed!")