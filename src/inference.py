"""
Inference module for animal disease classification predictions
"""
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Dict, List, Tuple, Union
import logging
import json

from config import Config
from data_preprocessing import DataPreprocessor

# Optional Grad-CAM import
try:
    from gradcam import GradCAMIntegration
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logger.warning("Grad-CAM functionality not available")

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class DiseasePredictor:
    """
    Disease prediction class for single images and batch predictions
    """
    
    def __init__(self, model_path: str, config: Config = None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model
            config: Configuration object
        """
        self.config = config or Config()
        self.model = None
        self.class_names = None
        self.preprocessor = DataPreprocessor(self.config)
        self.gradcam_integration = None
        
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load the trained model and class names.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            # Try to load class names from a separate file
            class_names_path = model_path.replace('.h5', '_classes.json')
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
            else:
                # Use default class names from config
                self.class_names = self.config.DISEASE_CATEGORIES
            
            logger.info(f"Model loaded successfully. Classes: {self.class_names}")
            
            # Initialize Grad-CAM if available
            if GRADCAM_AVAILABLE:
                try:
                    self.gradcam_integration = GradCAMIntegration(self.model, self.class_names)
                    logger.info("Grad-CAM integration initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Grad-CAM: {e}")
                    self.gradcam_integration = None
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_single_image(self, image_path: str, top_k: int = 3) -> Dict:
        """
        Predict disease for a single image.
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        try:
            # Load and preprocess image
            image = self.preprocessor.load_and_preprocess_image(image_path)
            if image is None:
                return {'error': 'Failed to load or preprocess image'}
            
            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(image_batch, verbose=0)[0]
            
            # Get top-k predictions
            top_indices = np.argsort(predictions)[::-1][:top_k]
            
            results = {
                'image_path': image_path,
                'predictions': [],
                'top_prediction': {
                    'class': self.class_names[top_indices[0]],
                    'confidence': float(predictions[top_indices[0]])
                }
            }
            
            # Add all top-k predictions
            for idx in top_indices:
                results['predictions'].append({
                    'class': self.class_names[idx],
                    'confidence': float(predictions[idx]),
                    'percentage': float(predictions[idx] * 100)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {str(e)}")
            return {'error': str(e)}
    
    def predict_batch(self, image_paths: List[str], top_k: int = 3) -> List[Dict]:
        """
        Predict diseases for a batch of images.
        
        Args:
            image_paths: List of image file paths
            top_k: Number of top predictions to return for each image
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict_single_image(image_path, top_k)
            results.append(result)
        
        return results
    
    def predict_from_directory(self, directory_path: str, top_k: int = 3) -> List[Dict]:
        """
        Predict diseases for all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            top_k: Number of top predictions to return for each image
            
        Returns:
            List of prediction dictionaries
        """
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = []
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(image_extensions):
                image_paths.append(os.path.join(directory_path, filename))
        
        logger.info(f"Found {len(image_paths)} images in {directory_path}")
        return self.predict_batch(image_paths, top_k)
    
    def predict_with_preprocessing_info(self, image_path: str) -> Dict:
        """
        Predict with additional preprocessing information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with predictions and preprocessing info
        """
        try:
            # Load original image for info
            original_image = Image.open(image_path)
            original_size = original_image.size
            
            # Make prediction
            prediction_result = self.predict_single_image(image_path)
            
            # Add preprocessing info
            prediction_result.update({
                'preprocessing_info': {
                    'original_size': original_size,
                    'target_size': self.config.IMAGE_SIZE,
                    'channels': self.config.CHANNELS,
                    'normalization': 'ImageNet' if hasattr(self.config, 'MEAN') else 'Standard'
                }
            })
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error in detailed prediction: {str(e)}")
            return {'error': str(e)}
    
    def get_prediction_explanation(self, image_path: str) -> Dict:
        """
        Get detailed explanation of the prediction process.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction explanation
        """
        result = self.predict_with_preprocessing_info(image_path)
        
        if 'error' in result:
            return result
        
        # Add explanation
        top_prediction = result['top_prediction']
        confidence = top_prediction['confidence']
        
        # Determine confidence level
        if confidence > 0.8:
            confidence_level = 'High'
            explanation = f"The model is very confident ({confidence:.1%}) that this is {top_prediction['class']}."
        elif confidence > 0.6:
            confidence_level = 'Medium'
            explanation = f"The model has moderate confidence ({confidence:.1%}) that this is {top_prediction['class']}."
        else:
            confidence_level = 'Low'
            explanation = f"The model has low confidence ({confidence:.1%}) in its prediction. Consider getting additional validation."
        
        result['explanation'] = {
            'confidence_level': confidence_level,
            'explanation': explanation,
            'recommendation': self._get_recommendation(top_prediction['class'], confidence)
        }
        
        return result
    
    def predict_with_gradcam(self, image_path: str, top_k: int = 3) -> Dict:
        """
        Get prediction with Grad-CAM visualizations.
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to include
            
        Returns:
            Dictionary with prediction and Grad-CAM results
        """
        if not self.gradcam_integration:
            return {
                'error': 'Grad-CAM not available. Please ensure the model supports Grad-CAM visualization.'
            }
        
        try:
            # Load and preprocess image
            image = self.preprocessor.load_and_preprocess_image(image_path)
            if image is None:
                return {'error': 'Failed to load or preprocess image'}
            
            # Get prediction with Grad-CAM
            gradcam_result = self.gradcam_integration.get_prediction_with_gradcam(
                image, top_k=top_k, include_all_heatmaps=True
            )
            
            # Add additional context
            result = {
                'image_path': image_path,
                'prediction': gradcam_result['prediction'],
                'all_predictions': gradcam_result['all_predictions'],
                'gradcam': gradcam_result['gradcam'],
                'explanation': self._get_gradcam_explanation(
                    gradcam_result['prediction']['predicted_class'],
                    gradcam_result['prediction']['confidence']
                )
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Grad-CAM prediction for {image_path}: {str(e)}")
            return {'error': str(e)}
    
    def _get_gradcam_explanation(self, predicted_class: str, confidence: float) -> Dict:
        """
        Get explanation for Grad-CAM visualization.
        
        Args:
            predicted_class: Predicted disease class
            confidence: Confidence score
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            'visual_explanation': 'The colored heatmap shows which parts of the image the AI model focused on when making its prediction.',
            'color_interpretation': {
                'red/hot_areas': 'Most important regions for the prediction',
                'blue/cool_areas': 'Less important regions',
                'intensity': 'Higher intensity indicates greater importance'
            },
            'prediction_confidence': self._get_confidence_explanation(confidence),
            'clinical_recommendation': self._get_recommendation(predicted_class, confidence)
        }
        
        return explanation
    
    def _get_confidence_explanation(self, confidence: float) -> Dict:
        """
        Get explanation for confidence level.
        
        Args:
            confidence: Confidence score
            
        Returns:
            Confidence explanation dictionary
        """
        if confidence > 0.9:
            level = 'Very High'
            description = 'The model is extremely confident in its prediction.'
            reliability = 'Very Reliable'
        elif confidence > 0.8:
            level = 'High'
            description = 'The model is very confident in its prediction.'
            reliability = 'Reliable'
        elif confidence > 0.7:
            level = 'Good'
            description = 'The model has good confidence in its prediction.'
            reliability = 'Generally Reliable'
        elif confidence > 0.6:
            level = 'Moderate'
            description = 'The model has moderate confidence in its prediction.'
            reliability = 'Moderately Reliable'
        elif confidence > 0.5:
            level = 'Low'
            description = 'The model has low confidence in its prediction.'
            reliability = 'Less Reliable'
        else:
            level = 'Very Low'
            description = 'The model has very low confidence in its prediction.'
            reliability = 'Unreliable'
        
        return {
            'level': level,
            'percentage': f"{confidence:.1%}",
            'description': description,
            'reliability': reliability,
            'recommendation': 'Consider additional diagnostic methods.' if confidence < 0.7 else 'Prediction appears reliable.'
        }
    
    def _get_recommendation(self, predicted_class: str, confidence: float) -> str:
        """
        Get recommendation based on prediction.
        
        Args:
            predicted_class: Predicted disease class
            confidence: Confidence score
            
        Returns:
            Recommendation string
        """
        recommendations = {
            'healthy': 'The animal appears healthy. Continue regular monitoring and preventive care.',
            'bacterial_infection': 'Bacterial infection detected. Consult a veterinarian for appropriate antibiotic treatment.',
            'viral_infection': 'Viral infection detected. Supportive care and isolation may be needed. Consult a veterinarian.',
            'fungal_infection': 'Fungal infection detected. Antifungal treatment may be required. Consult a veterinarian.',
            'parasitic_infection': 'Parasitic infection detected. Antiparasitic treatment is recommended. Consult a veterinarian.',
            'nutritional_deficiency': 'Nutritional deficiency detected. Review diet and consider supplements. Consult a veterinarian.',
            'genetic_disorder': 'Genetic disorder detected. Specialized care may be needed. Consult a specialist veterinarian.'
        }
        
        base_recommendation = recommendations.get(predicted_class, 'Consult a veterinarian for proper diagnosis and treatment.')
        
        if confidence < 0.6:
            base_recommendation += " Note: Low confidence in prediction - consider additional diagnostic tests."
        
        return base_recommendation

def save_predictions_to_file(predictions: List[Dict], output_path: str):
    """
    Save predictions to a JSON file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save the predictions
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")

def load_predictions_from_file(input_path: str) -> List[Dict]:
    """
    Load predictions from a JSON file.
    
    Args:
        input_path: Path to the predictions file
        
    Returns:
        List of prediction dictionaries
    """
    try:
        with open(input_path, 'r') as f:
            predictions = json.load(f)
        logger.info(f"Predictions loaded from {input_path}")
        return predictions
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}")
        return []

if __name__ == "__main__":
    # Example usage
    config = Config()
    
    # Check if we have a trained model
    if os.path.exists(config.BEST_MODEL_PATH):
        # Initialize predictor
        predictor = DiseasePredictor(config.BEST_MODEL_PATH, config)
        
        # Example: predict a single image (if exists)
        test_image_dir = os.path.join(config.TEST_DATA_DIR, 'healthy')  # Example class
        if os.path.exists(test_image_dir):
            test_images = [f for f in os.listdir(test_image_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            if test_images:
                test_image_path = os.path.join(test_image_dir, test_images[0])
                result = predictor.get_prediction_explanation(test_image_path)
                
                print("Prediction Result:")
                print(json.dumps(result, indent=2))
            else:
                print("No test images found.")
        else:
            print("Test directory not found.")
    else:
        print("No trained model found. Please train a model first.")