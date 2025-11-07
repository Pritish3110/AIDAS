"""
Evaluation and metrics module for animal disease classification
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import tensorflow as tf
from typing import Dict, List, Tuple
import logging

from config import Config
from data_preprocessing import DataPreprocessor

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation with various metrics and visualizations
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
    def evaluate_model(self, model: tf.keras.Model, test_dataset: tf.data.Dataset, 
                      class_names: List[str]) -> Dict:
        """
        Comprehensive evaluation of a trained model.
        
        Args:
            model: Trained Keras model
            test_dataset: Test dataset
            class_names: List of class names
            
        Returns:
            Dictionary containing various evaluation metrics
        """
        logger.info("Starting model evaluation...")
        
        # Get predictions
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for batch_images, batch_labels in test_dataset:
            batch_predictions = model.predict(batch_images, verbose=0)
            y_pred_proba.extend(batch_predictions)
            y_pred.extend(np.argmax(batch_predictions, axis=1))
            y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'per_class_metrics': {
                'precision': per_class_precision.tolist(),
                'recall': per_class_recall.tolist(),
                'f1_score': per_class_f1.tolist(),
                'support': per_class_support.tolist()
            },
            'class_names': class_names
        }
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        return results, y_true, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            save_path: str = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history: Dict, save_path: str = None, 
                            figsize: Tuple[int, int] = (15, 5)):
        """
        Plot training history curves.
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Loss plot
        axes[0].plot(history['loss'], label='Training Loss', marker='o', markersize=3)
        axes[0].plot(history['val_loss'], label='Validation Loss', marker='s', markersize=3)
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['accuracy'], label='Training Accuracy', marker='o', markersize=3)
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', marker='s', markersize=3)
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history:
            axes[2].plot(history['lr'], label='Learning Rate', marker='o', markersize=3)
            axes[2].set_title('Learning Rate')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, dataset_dir: str, save_path: str = None,
                               figsize: Tuple[int, int] = (12, 6)):
        """
        Plot class distribution in the dataset.
        
        Args:
            dataset_dir: Directory containing the dataset
            save_path: Path to save the plot
            figsize: Figure size
        """
        preprocessor = DataPreprocessor()
        dataset_info = preprocessor.get_dataset_info(dataset_dir)
        
        class_names = list(dataset_info['class_distribution'].keys())
        class_counts = list(dataset_info['class_distribution'].values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        bars = ax1.bar(class_names, class_counts, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_title('Class Distribution')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Images')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(class_counts),
                    f'{int(height)}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(class_counts, labels=class_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution (Percentage)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, results: Dict, output_dir: str):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            output_dir: Directory to save the report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Animal Disease Classification - Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .class-metrics {{ background-color: #e8f4fd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>Animal Disease Classification - Evaluation Report</h1>
            
            <h2>Overall Metrics</h2>
            <div class="metric"><strong>Accuracy:</strong> {results['accuracy']:.4f}</div>
            <div class="metric"><strong>Precision:</strong> {results['precision']:.4f}</div>
            <div class="metric"><strong>Recall:</strong> {results['recall']:.4f}</div>
            <div class="metric"><strong>F1-Score:</strong> {results['f1_score']:.4f}</div>
            
            <h2>Per-Class Metrics</h2>
            <table>
                <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
        """
        
        for i, class_name in enumerate(results['class_names']):
            html_content += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{results['per_class_metrics']['precision'][i]:.4f}</td>
                    <td>{results['per_class_metrics']['recall'][i]:.4f}</td>
                    <td>{results['per_class_metrics']['f1_score'][i]:.4f}</td>
                    <td>{results['per_class_metrics']['support'][i]}</td>
                </tr>
            """
        
        html_content += """
            </table>
            </body>
            </html>
        """
        
        report_path = os.path.join(output_dir, 'evaluation_report.html')
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to {report_path}")

def evaluate_trained_model(model_path: str, test_data_dir: str) -> Dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to the saved model
        test_data_dir: Directory containing test data
        
    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Prepare test dataset
    preprocessor = DataPreprocessor()
    test_dataset = preprocessor.create_tensorflow_dataset(test_data_dir, augment=False)
    
    # Evaluate
    evaluator = ModelEvaluator()
    results, y_true, y_pred, y_pred_proba = evaluator.evaluate_model(
        model, test_dataset, test_dataset.class_names
    )
    
    return results

if __name__ == "__main__":
    # Example usage
    config = Config()
    
    # Check if we have a trained model and test data
    if os.path.exists(config.BEST_MODEL_PATH) and os.path.exists(config.TEST_DATA_DIR):
        results = evaluate_trained_model(config.BEST_MODEL_PATH, config.TEST_DATA_DIR)
        
        # Generate evaluation report
        evaluator = ModelEvaluator()
        evaluator.generate_evaluation_report(results, config.MODELS_DIR)
        
        logger.info("Evaluation completed successfully!")
    else:
        logger.info("No trained model or test data found. Please train a model first.")