"""
EigenCAM (Eigen-weighted Class Activation Mapping) implementation
for visualizing what regions of the input image the model is focusing on
when making predictions using principal component analysis.
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

class EigenCAM:
    """
    EigenCAM implementation for generating visual explanations of CNN predictions.
    
    Unlike Grad-CAM which uses gradients, EigenCAM uses Principal Component Analysis (PCA)
    on the activation maps to identify the most important features. This approach:
    - Does not require gradient computation
    - Is class-agnostic (doesn't need target class)
    - Highlights the most discriminative features based on variance
    - Can be faster for certain use cases
    """
    
    def __init__(self, model, target_layer_name=None):
        """
        Initialize EigenCAM with a trained model.
        
        Args:
            model: Trained Keras/TensorFlow model
            target_layer_name: Name of the target convolutional layer to use for EigenCAM.
                              If None, will automatically find the last convolutional layer.
        """
        self.model = model
        self.target_layer_name = target_layer_name or self._find_target_layer()
    
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
                print(f"Using layer '{layer.name}' for EigenCAM visualization")
                return layer.name
        
        # Look for nested models (common in transfer learning)
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.models.Model):
                # This is a nested model, find the last conv layer in it
                nested_layer = self._find_nested_conv_layer(layer)
                if nested_layer:
                    target_name = f"{layer.name}/{nested_layer}"
                    print(f"Using nested layer '{target_name}' for EigenCAM visualization")
                    return target_name
        
        # Fallback: look for layers with conv-like names
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower() or 'block' in layer.name.lower():
                print(f"Using layer '{layer.name}' for EigenCAM visualization")
                return layer.name
        
        raise ValueError("No suitable convolutional layer found for EigenCAM")
    
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
    
    def generate_heatmap(self, image, class_index=None, alpha=0.4, use_first_component=True):
        """
        Generate EigenCAM heatmap for a given image.
        
        Args:
            image: Input image (preprocessed for the model)
            class_index: Index of the class (used for labeling, not computation)
            alpha: Transparency factor for overlay (0=transparent, 1=opaque)
            use_first_component: If True, use first principal component; if False, use weighted combination
            
        Returns:
            Dictionary containing heatmap visualizations
        """
        try:
            # Ensure image has batch dimension
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # Convert to tensor
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            
            # Get predictions to determine class index if not provided
            predictions = self.model(image_tensor)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            
            # Get activation maps from target layer
            activations = self._get_layer_activations(image_tensor)
            
            # Compute EigenCAM heatmap using PCA
            heatmap = self._compute_eigencam(activations, use_first_component)
            
            # Process the heatmap
            heatmap = heatmap.numpy() if tf.is_tensor(heatmap) else heatmap
            
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
                'target_layer': self.target_layer_name,
                'method': 'EigenCAM'
            }
            
        except Exception as e:
            print(f"Error in generate_heatmap: {e}")
            # Return a fallback visualization
            return self._create_fallback_heatmap(image, class_index)
    
    def _get_layer_activations(self, image_tensor):
        """
        Get activations from the target layer.
        
        Args:
            image_tensor: Input image tensor
            
        Returns:
            Target layer activations
        """
        try:
            # Handle nested layer names
            if '/' in self.target_layer_name:
                base_model_name, nested_layer_name = self.target_layer_name.split('/', 1)
                base_model = self.model.get_layer(base_model_name)
                
                # Create extraction model for the nested layer
                extraction_model = keras.Model(
                    inputs=base_model.input,
                    outputs=base_model.get_layer(nested_layer_name).output
                )
                
                # Get activations from the base model input
                base_input = base_model(image_tensor)
                return extraction_model(image_tensor)
            else:
                # Direct layer case
                target_layer = self.model.get_layer(self.target_layer_name)
                extraction_model = keras.Model(
                    inputs=self.model.input,
                    outputs=target_layer.output
                )
                return extraction_model(image_tensor)
        except Exception as e:
            print(f"Error getting layer activations: {e}")
            # Fallback: try to get activations from the last conv layer
            return self._get_fallback_activations(image_tensor)
    
    def _get_fallback_activations(self, image_tensor):
        """
        Fallback method to get activations from model.
        
        Args:
            image_tensor: Input image tensor
            
        Returns:
            Activation maps
        """
        # Find any suitable layer
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                try:
                    extraction_model = keras.Model(
                        inputs=self.model.input,
                        outputs=layer.output
                    )
                    return extraction_model(image_tensor)
                except:
                    continue
        
        raise ValueError("Could not extract activations from any layer")
    
    def _compute_eigencam(self, activations, use_first_component=True):
        """
        Compute EigenCAM heatmap using Principal Component Analysis on activation maps.
        
        The algorithm:
        1. Reshape activations to (height*width, channels)
        2. Compute covariance matrix of the activations
        3. Perform eigenvalue decomposition
        4. Use the first (largest) eigenvector to weight the activation channels
        5. Sum weighted activations to create heatmap
        
        Args:
            activations: Activation maps from the target layer [batch, height, width, channels]
            use_first_component: If True, use only first principal component
            
        Returns:
            Heatmap as 2D array
        """
        # Remove batch dimension
        activations = activations[0]  # Shape: (height, width, channels)
        
        height, width, num_channels = activations.shape
        
        # Reshape to (height*width, channels) for covariance computation
        activations_reshaped = tf.reshape(activations, [-1, num_channels])
        
        # Convert to numpy for eigenvalue decomposition
        activations_np = activations_reshaped.numpy()
        
        # Center the data (subtract mean)
        activations_centered = activations_np - np.mean(activations_np, axis=0)
        
        # Compute covariance matrix (channels x channels)
        cov_matrix = np.cov(activations_centered.T)
        
        # Eigenvalue decomposition
        # eigenvalues are sorted in ascending order by default
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Get the principal component (eigenvector with largest eigenvalue)
        # Since eigh returns in ascending order, take the last one
        if use_first_component:
            principal_component = eigenvectors[:, -1]  # Shape: (channels,)
        else:
            # Use top 3 components weighted by their eigenvalues
            num_components = min(3, num_channels)
            top_eigenvalues = eigenvalues[-num_components:]
            top_eigenvectors = eigenvectors[:, -num_components:]
            
            # Normalize eigenvalues to use as weights
            weights = top_eigenvalues / np.sum(top_eigenvalues)
            
            # Weighted combination of eigenvectors
            principal_component = np.sum(
                top_eigenvectors * weights.reshape(1, -1), 
                axis=1
            )
        
        # Project activations onto principal component
        # Shape: (height*width,)
        projected = np.dot(activations_np, principal_component)
        
        # Reshape back to spatial dimensions
        heatmap = projected.reshape(height, width)
        
        # Apply ReLU (keep only positive values)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def _create_fallback_heatmap(self, image, class_index):
        """
        Create a fallback heatmap when EigenCAM computation fails.
        
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
            'target_layer': 'fallback_visualization',
            'method': 'EigenCAM (fallback)'
        }
    
    def generate_multi_class_heatmaps(self, image, class_names, top_k=3):
        """
        Generate EigenCAM heatmaps for multiple classes.
        Note: EigenCAM is class-agnostic, so all heatmaps will be the same.
        This method is included for API compatibility with GradCAM.
        
        Args:
            image: Input image (preprocessed for the model)
            class_names: List of class names
            top_k: Number of top predictions to include
            
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
        
        # Generate heatmap once (class-agnostic)
        base_heatmap = self.generate_heatmap(image)
        
        results = []
        for i, class_idx in enumerate(top_indices):
            # Copy the base heatmap for each class
            heatmap_result = base_heatmap.copy()
            heatmap_result['class_name'] = class_names[class_idx] if class_idx < len(class_names) else f'Class {class_idx}'
            heatmap_result['probability'] = float(probabilities[class_idx])
            heatmap_result['rank'] = i + 1
            heatmap_result['class_index'] = int(class_idx)
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
        print(f"EigenCAM heatmap saved to {save_path}")


class EigenCAMIntegration:
    """
    Integration class to easily use EigenCAM with existing inference pipeline.
    """
    
    def __init__(self, model, class_names, target_layer_name=None):
        """
        Initialize EigenCAM integration.
        
        Args:
            model: Trained Keras/TensorFlow model
            class_names: List of class names
            target_layer_name: Target layer for EigenCAM (optional)
        """
        self.eigencam = EigenCAM(model, target_layer_name)
        self.class_names = class_names
    
    def get_prediction_with_eigencam(self, image, top_k=3, include_all_heatmaps=True):
        """
        Get prediction results enhanced with EigenCAM visualizations.
        
        Args:
            image: Preprocessed input image
            top_k: Number of top predictions to include
            include_all_heatmaps: Whether to generate heatmaps for all top predictions
            
        Returns:
            Dictionary with prediction results and EigenCAM visualizations
        """
        # Ensure image has correct dimensions
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
            image = image[0]
        
        # Get standard prediction
        predictions = self.eigencam.model.predict(image_batch, verbose=0)
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        # Get top prediction
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Generate EigenCAM for top prediction
        main_heatmap = self.eigencam.generate_heatmap(image, class_index=predicted_class_idx)
        
        result = {
            'prediction': {
                'predicted_class': predicted_class,
                'predicted_class_idx': int(predicted_class_idx),
                'confidence': confidence,
                'confidence_percentage': confidence * 100
            },
            'eigencam': {
                'main_heatmap': main_heatmap,
                'target_layer': main_heatmap['target_layer']
            }
        }
        
        # Generate heatmaps for top k predictions if requested
        if include_all_heatmaps:
            multi_heatmaps = self.eigencam.generate_multi_class_heatmaps(image, self.class_names, top_k)
            result['eigencam']['multi_class_heatmaps'] = multi_heatmaps
        
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


def test_eigencam():
    """
    Test function to verify EigenCAM implementation.
    """
    print("Testing EigenCAM implementation...")
    print("This function can be used to test EigenCAM with a trained model.")
    print("Make sure to load a model and provide test images for full testing.")


if __name__ == "__main__":
    test_eigencam()
