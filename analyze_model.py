#!/usr/bin/env python3
"""
Analyze model architecture to find suitable layers for Grad-CAM
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, 'src')

def analyze_model_architecture(model_path):
    """Analyze the model architecture and find suitable layers for Grad-CAM"""
    print(f"🔍 Analyzing model: {model_path}")
    print("=" * 60)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total layers: {len(model.layers)}")
    
    print("\n📋 Layer Analysis:")
    print("-" * 60)
    
    suitable_layers = []
    
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        layer_name = layer.name
        
        # Check if layer has filters (Conv2D layers)
        has_filters = hasattr(layer, 'filters')
        
        # Check if it's a nested model (like transfer learning base)
        is_model = isinstance(layer, tf.keras.models.Model)
        
        # Check for conv-like patterns in name
        has_conv_name = 'conv' in layer_name.lower() or 'block' in layer_name.lower()
        
        print(f"Layer {i:2d}: {layer_name:<25} | {layer_type:<20} | Filters: {has_filters} | Model: {is_model}")
        
        if has_filters:
            suitable_layers.append((i, layer_name, 'direct_conv'))
        elif is_model:
            # This is likely a nested model (like EfficientNet, ResNet base)
            print(f"         -> Nested model detected, analyzing...")
            nested_suitable = analyze_nested_model(layer, f"  Layer {i}")
            suitable_layers.extend([(i, f"{layer_name}/{sub_name}", 'nested') for sub_idx, sub_name in nested_suitable])
        elif has_conv_name:
            suitable_layers.append((i, layer_name, 'conv_named'))
    
    print("\n🎯 Suitable layers for Grad-CAM:")
    print("-" * 40)
    
    if suitable_layers:
        for idx, name, layer_type in suitable_layers[-10:]:  # Show last 10
            print(f"  {idx}: {name} ({layer_type})")
        
        # Recommend the best layer
        best_layer = suitable_layers[-1]  # Usually the last conv layer
        print(f"\n✅ Recommended layer: '{best_layer[1]}'")
        return best_layer[1]
    else:
        print("  ❌ No suitable layers found")
        return None

def analyze_nested_model(nested_model, prefix=""):
    """Analyze nested models to find convolutional layers"""
    suitable = []
    
    for i, layer in enumerate(nested_model.layers[-20:]):  # Check last 20 layers
        if hasattr(layer, 'filters') or 'conv' in layer.name.lower():
            suitable.append((i, layer.name))
            if len(suitable) <= 5:  # Show only first few
                print(f"{prefix}   -> {layer.name} ({type(layer).__name__})")
    
    return suitable

if __name__ == "__main__":
    # Find and analyze the first available model
    model_paths = [
        'models/enhanced_90plus_final.h5',
        'models/enhanced_final_90plus.h5',
        'models/ultimate_final.h5',
        'models/resnet50_final.h5',
        'models/simple_custom_model.h5'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            recommended_layer = analyze_model_architecture(model_path)
            
            if recommended_layer:
                print(f"\n💡 To fix Grad-CAM, use this layer name: '{recommended_layer}'")
                print("\nThis layer should be used in the GradCAM initialization.")
            break
    else:
        print("No models found to analyze!")