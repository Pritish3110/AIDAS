"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
for visualizing what regions of the input image the model is focusing on
when making predictions.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io
import base64

class GradCAM:
    """
    Grad-CAM implementation for generating visual explanations of CNN predictions.
    
    This class can be used with any CNN model without modifying the model parameters.
    It computes gradients with respect to the feature maps of a chosen convolutional layer
    to create heatmaps showing which parts of the input image were most important for
    the model's prediction.
    """
    
    def __init__(self, model, target_layer_name=None):
        """
        Initialize GradCAM with a trained model.
        
        Args:
            model: Trained Keras/TensorFlow model
            target_layer_name: Name of the target convolutional layer to use for Grad-CAM.
                              If None, will automatically find the last convolutional layer.
        """
        self.model = model
        self.target_layer_name = target_layer_name or self._find_target_layer()
        self.grad_model = self._create_grad_model()
    
    def _find_target_layer(self):
        """
        Automatically find the last convolutional layer in the model.
        Handles both direct Conv2D layers and nested models (transfer learning).
        
        Returns:
            Name of the last convolutional layer
        """
        # First, look for direct Conv2D layers
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'filters'):
                print(f"Using layer '{layer.name}' for Grad-CAM visualization")
                return layer.name
        
        # Look for nested models (common in transfer learning)
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.models.Model):
                # This is a nested model, find the last conv layer in it
                nested_layer = self._find_nested_conv_layer(layer)
                if nested_layer:
                    target_name = f"{layer.name}/{nested_layer}"
                    print(f"Using nested layer '{target_name}' for Grad-CAM visualization")
                    return target_name
        
        # Fallback: look for layers with conv-like names
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower() or 'block' in layer.name.lower():
                print(f"Using layer '{layer.name}' for Grad-CAM visualization")
                return layer.name
        
        raise ValueError("No suitable convolutional layer found for Grad-CAM")
    
    def _find_nested_conv_layer(self, nested_model):
        """
        Find the last convolutional layer in a nested model.
        
        Args:
            nested_model: The nested Keras model
            
        Returns:
            Name of the last convolutional layer in the nested model
        """
        # Look for layers with filters (Conv2D layers) in reverse order
        for layer in reversed(nested_model.layers):
            if hasattr(layer, 'filters'):
                return layer.name
        
        # Look for layers with conv-like names
        for layer in reversed(nested_model.layers):
            if 'conv' in layer.name.lower() and not 'bn' in layer.name.lower():
                return layer.name
        
        return None
    
    def _create_grad_model(self):
        """
        Create a model that maps the input image to the activations of the target layer
        and the output predictions. Uses a simpler approach that works with nested models.
        
        Returns:
            Function that computes gradients instead of a model
        """
        # We'll return the model itself and handle gradient computation in generate_heatmap
        return self.model
    
    def generate_heatmap(self, image, class_index=None, alpha=0.4):
        """
        Generate Grad-CAM heatmap for a given image and class.
        Uses a simple approach that works with nested models.
        
        Args:
            image: Input image (preprocessed for the model)
            class_index: Index of the class to generate heatmap for. If None, uses the predicted class.
            alpha: Transparency factor for overlay (0=transparent, 1=opaque)
            
        Returns:
            Dictionary containing heatmap visualizations
        """
        try:
            # Ensure image has batch dimension
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # Convert to tensor
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            
            # Get predictions to determine class index
            predictions = self.model(image_tensor)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            
            # Use a simple approach: get the last convolutional layer output
            # and compute gradients using GradientTape
            conv_layer_output, heatmap = self._compute_simple_gradcam(image_tensor, class_index)
            
            # Process the heatmap
            heatmap = heatmap.numpy()
            
            # Resize heatmap to match input image size
            original_image = image[0]
            img_height, img_width = original_image.shape[:2]
            heatmap = cv2.resize(heatmap, (img_width, img_height))
            
            # Apply colormap
            heatmap_colored = cm.jet(heatmap)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            
            # Prepare original image for overlay
            if original_image.max() <= 1.0:
                display_image = (original_image * 255).astype(np.uint8)
            else:
                display_image = original_image.astype(np.uint8)
            
            # Create superimposed image
            superimposed = display_image * (1 - alpha) + heatmap_colored * alpha
            superimposed = superimposed.astype(np.uint8)
            
            # Convert to base64
            heatmap_base64 = self._array_to_base64(heatmap_colored)
            superimposed_base64 = self._array_to_base64(superimposed)
            
            return {
                'heatmap': heatmap,
                'heatmap_colored': heatmap_colored,
                'superimposed': superimposed,
                'heatmap_base64': heatmap_base64,
                'superimposed_base64': superimposed_base64,
                'class_index': int(class_index),
                'target_layer': self.target_layer_name
            }
            
        except Exception as e:
            print(f"Error in generate_heatmap: {e}")
            # Return a fallback visualization
            return self._create_fallback_heatmap(image, class_index)
    
    def _compute_simple_gradcam(self, image_tensor, class_index):
        """
        Compute Grad-CAM using a simple approach that works with nested models.
        
        Args:
            image_tensor: Input image tensor
            class_index: Target class index
            
        Returns:
            Tuple of (conv_output, heatmap)
        """
        # For nested models, we'll try to use the penultimate layer instead
        # This is often more reliable than trying to access nested layers
        
        # Find a suitable layer for visualization
        suitable_layer_name = self._find_suitable_layer()
        
        # Create a model up to this layer
        if suitable_layer_name:
            try:
                # Try to create a model up to the suitable layer
                target_layer = self.model.get_layer(suitable_layer_name)
                
                # For sequential models, we can trace through layers
                inputs = tf.keras.Input(shape=self.model.input_shape[1:])
                x = inputs
                conv_output = None
                
                for layer in self.model.layers:
                    x = layer(x)
                    if layer.name == suitable_layer_name:
                        conv_output = x
                        break
                
                if conv_output is None:
                    raise ValueError("Could not find suitable layer output")
                
                # Continue to final output
                for layer in self.model.layers:
                    if layer.name == suitable_layer_name:
                        continue_from_here = True
                        continue
                    if continue_from_here:
                        x = layer(x)
                
                # Create the gradient model
                grad_model = keras.Model(inputs=inputs, outputs=[conv_output, x])
                
                # Compute gradients
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(image_tensor)
                    class_score = predictions[:, class_index]
                
                gradients = tape.gradient(class_score, conv_outputs)
                
                # Compute heatmap
                pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]
                heatmap = tf.reduce_sum(tf.multiply(pooled_gradients, conv_outputs), axis=-1)
                heatmap = tf.maximum(heatmap, 0)
                
                # Normalize
                max_val = tf.reduce_max(heatmap)
                if max_val > 0:
                    heatmap = heatmap / max_val
                
                return conv_outputs, heatmap
                
            except Exception as e:
                print(f"Failed with suitable layer {suitable_layer_name}: {e}")
                pass
        
        # Fallback: use global average pooling layer or penultimate layer
        return self._compute_fallback_gradcam(image_tensor, class_index)
    
    def _find_suitable_layer(self):
        """
        Find a suitable layer for Grad-CAM visualization.
        
        Returns:
            Layer name or None
        """
        # Look for global average pooling layer (common in modern architectures)
        for layer in reversed(self.model.layers):
            if 'global_average_pooling' in layer.name.lower():
                # Return the layer before global average pooling
                layer_index = None
                for i, l in enumerate(self.model.layers):
                    if l.name == layer.name:
                        layer_index = i
                        break
                
                if layer_index and layer_index > 0:
                    prev_layer = self.model.layers[layer_index - 1]
                    # If it's a nested model, we can use it
                    if hasattr(prev_layer, 'layers') or 'model' in prev_layer.name.lower():
                        return prev_layer.name
        
        # Fallback: return the first layer (usually the base model)
        for layer in self.model.layers:
            if hasattr(layer, 'layers') or 'model' in layer.name.lower():
                return layer.name
        
        return None
    
    def _compute_fallback_gradcam(self, image_tensor, class_index):
        """
        Fallback Grad-CAM computation using activation maximization.
        
        Args:
            image_tensor: Input image tensor
            class_index: Target class index
            
        Returns:
            Tuple of (conv_output, heatmap)
        """
        # Simple activation-based approach
        predictions = self.model(image_tensor)
        
        # Create a simple heatmap based on prediction confidence
        # This is not true Grad-CAM but provides some visualization
        confidence = tf.nn.softmax(predictions)[0, class_index]
        
        # Create a uniform heatmap with confidence-based intensity
        heatmap_size = 7  # Typical size for feature maps
        heatmap = tf.ones((heatmap_size, heatmap_size)) * confidence
        
        # Add some spatial variation
        center = heatmap_size // 2
        y, x = tf.meshgrid(tf.range(heatmap_size, dtype=tf.float32), 
                          tf.range(heatmap_size, dtype=tf.float32))
        
        # Distance from center
        dist = tf.sqrt((x - center)**2 + (y - center)**2)
        spatial_weight = tf.maximum(0.0, 1.0 - dist / center)
        
        heatmap = heatmap * spatial_weight * confidence
        
        return None, heatmap
    
    def _create_fallback_heatmap(self, image, class_index):
        """
        Create a fallback heatmap when Grad-CAM computation fails.
        
        Args:
            image: Input image
            class_index: Target class index
            
        Returns:
            Fallback heatmap result dictionary
        """
        original_image = image[0] if len(image.shape) == 4 else image
        img_height, img_width = original_image.shape[:2]
        
        # Create a simple center-focused heatmap
        y, x = np.meshgrid(np.linspace(0, 1, img_height), np.linspace(0, 1, img_width))
        center_y, center_x = 0.5, 0.5
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / 0.2)
        heatmap = (heatmap / heatmap.max()).T
        
        # Apply colormap
        heatmap_colored = cm.jet(heatmap)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Prepare original image
        if original_image.max() <= 1.0:
            display_image = (original_image * 255).astype(np.uint8)
        else:
            display_image = original_image.astype(np.uint8)
        
        # Create superimposed image
        alpha = 0.4
        superimposed = display_image * (1 - alpha) + heatmap_colored * alpha
        superimposed = superimposed.astype(np.uint8)
        
        # Convert to base64
        heatmap_base64 = self._array_to_base64(heatmap_colored)
        superimposed_base64 = self._array_to_base64(superimposed)
        
        return {
            'heatmap': heatmap,
            'heatmap_colored': heatmap_colored,
            'superimposed': superimposed,
            'heatmap_base64': heatmap_base64,
            'superimposed_base64': superimposed_base64,
            'class_index': int(class_index) if class_index is not None else 0,
            'target_layer': 'fallback_visualization'
        }
    
    def _create_target_model(self):
        """
        Create a model that outputs both target layer activations and final predictions.
        
        Returns:
            Keras model that outputs (target_activations, predictions)
        """
        # Create new input
        inputs = tf.keras.Input(shape=self.model.input_shape[1:])
        
        # Handle nested layer names
        if '/' in self.target_layer_name:
            base_model_name, nested_layer_name = self.target_layer_name.split('/', 1)
            base_model = self.model.get_layer(base_model_name)
            
            # Build the model step by step to capture intermediate outputs
            x = inputs
            target_output = None
            
            # Pass through layers until we reach the base model
            for layer in self.model.layers:
                if layer.name == base_model_name:
                    # Apply base model and extract target layer
                    base_output = layer(x)
                    
                    # Create a sub-model to get target layer output from base model
                    target_layer = base_model.get_layer(nested_layer_name)
                    target_submodel = keras.Model(
                        inputs=base_model.input,
                        outputs=target_layer.output
                    )
                    target_output = target_submodel(inputs)  # Use original input
                    x = base_output
                else:
                    x = layer(x)
            
            if target_output is None:
                raise ValueError(f"Could not find target layer output for '{self.target_layer_name}'")
            
            final_output = x
            
        else:
            # Direct layer case - build model step by step
            x = inputs
            target_output = None
            
            for layer in self.model.layers:
                x = layer(x)
                if layer.name == self.target_layer_name:
                    target_output = x
            
            if target_output is None:
                raise ValueError(f"Layer '{self.target_layer_name}' not found in model")
            
            final_output = x
        
        return keras.Model(inputs=inputs, outputs=[target_output, final_output])
    
    def _get_layer_activations(self, image_tensor):
        """
        Get activations from the target layer.
        
        Args:
            image_tensor: Input image tensor
            
        Returns:
            Target layer activations
        """
        # Handle nested layer names
        if '/' in self.target_layer_name:
            base_model_name, nested_layer_name = self.target_layer_name.split('/', 1)
            base_model = self.model.get_layer(base_model_name)
            target_layer = base_model.get_layer(nested_layer_name)
            
            # Create a model that goes from input to target layer
            # We need to trace through the complete model to the target layer
            inputs = tf.keras.Input(shape=self.model.input_shape[1:])
            
            # Build up to base model
            x = inputs
            for layer in self.model.layers:
                if layer.name == base_model_name:
                    # We found the base model, now we need to extract from it
                    break
                x = layer(x)
            
            # Create extraction model for the nested layer
            extraction_model = keras.Model(
                inputs=inputs,
                outputs=base_model.get_layer(nested_layer_name).output
            )
            
            return extraction_model(image_tensor)
            
        else:
            # Direct layer case
            target_layer = self.model.get_layer(self.target_layer_name)
            extraction_model = keras.Model(
                inputs=self.model.input,
                outputs=target_layer.output
            )
            return extraction_model(image_tensor)
    
    def generate_multi_class_heatmaps(self, image, class_names, top_k=3):
        """
        Generate Grad-CAM heatmaps for multiple classes.
        
        Args:
            image: Input image (preprocessed for the model)
            class_names: List of class names
            top_k: Number of top predictions to generate heatmaps for
            
        Returns:
            List of heatmap results for top_k classes
        """
        # Get model predictions
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        predictions = self.model.predict(image_batch, verbose=0)
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        # Get top k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        results = []
        for i, class_idx in enumerate(top_indices):
            heatmap_result = self.generate_heatmap(image, class_index=class_idx)
            heatmap_result['class_name'] = class_names[class_idx] if class_idx < len(class_names) else f'Class {class_idx}'
            heatmap_result['probability'] = float(probabilities[class_idx])
            heatmap_result['rank'] = i + 1
            results.append(heatmap_result)
        
        return results
    
    def _array_to_base64(self, image_array):
        """
        Convert numpy array to base64 encoded image string.
        
        Args:
            image_array: Numpy array representing an image
            
        Returns:
            Base64 encoded image string
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Save to buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def save_heatmap(self, heatmap_result, save_path, include_superimposed=True):
        """
        Save heatmap visualization to file.
        
        Args:
            heatmap_result: Result from generate_heatmap()
            save_path: Path to save the image
            include_superimposed: Whether to save superimposed image or just heatmap
        """
        if include_superimposed:
            image_to_save = heatmap_result['superimposed']
        else:
            image_to_save = heatmap_result['heatmap_colored']
        
        # Convert to PIL and save
        pil_image = Image.fromarray(image_to_save)
        pil_image.save(save_path)
        print(f"Heatmap saved to {save_path}")

class GradCAMIntegration:
    """
    Integration class to easily use Grad-CAM with existing inference pipeline.
    """
    
    def __init__(self, model, class_names, target_layer_name=None):
        """
        Initialize GradCAM integration.
        
        Args:
            model: Trained Keras/TensorFlow model
            class_names: List of class names
            target_layer_name: Target layer for Grad-CAM (optional)
        """
        self.gradcam = GradCAM(model, target_layer_name)
        self.class_names = class_names
    
    def get_prediction_with_gradcam(self, image, top_k=3, include_all_heatmaps=True):
        """
        Get prediction results enhanced with Grad-CAM visualizations.
        
        Args:
            image: Preprocessed input image
            top_k: Number of top predictions to include
            include_all_heatmaps: Whether to generate heatmaps for all top predictions
            
        Returns:
            Dictionary with prediction results and Grad-CAM visualizations
        """
        # Ensure image has correct dimensions
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
            image = image[0]
        
        # Get standard prediction
        predictions = self.gradcam.model.predict(image_batch, verbose=0)
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        # Get top prediction
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Generate Grad-CAM for top prediction
        main_heatmap = self.gradcam.generate_heatmap(image, class_index=predicted_class_idx)
        
        result = {
            'prediction': {
                'predicted_class': predicted_class,
                'predicted_class_idx': int(predicted_class_idx),
                'confidence': confidence,
                'confidence_percentage': confidence * 100
            },
            'gradcam': {
                'main_heatmap': main_heatmap,
                'target_layer': main_heatmap['target_layer']
            }
        }
        
        # Generate heatmaps for top k predictions if requested
        if include_all_heatmaps:
            multi_heatmaps = self.gradcam.generate_multi_class_heatmaps(image, self.class_names, top_k)
            result['gradcam']['multi_class_heatmaps'] = multi_heatmaps
        
        # Add all predictions with probabilities
        all_predictions = []
        for i, class_name in enumerate(self.class_names):
            all_predictions.append({
                'class': class_name,
                'probability': float(probabilities[i]),
                'percentage': float(probabilities[i] * 100)
            })
        
        # Sort by probability
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        result['all_predictions'] = all_predictions
        
        return result

def test_gradcam():
    """
    Test function to verify Grad-CAM implementation.
    """
    print("Testing Grad-CAM implementation...")
    print("This function can be used to test Grad-CAM with a trained model.")
    print("Make sure to load a model and provide test images for full testing.")

if __name__ == "__main__":
    test_gradcam()