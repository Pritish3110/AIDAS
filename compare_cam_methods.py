"""
Comprehensive Comparison Script: GradCAM vs EigenCAM
This script compares both explainability methods on the same images and generates
side-by-side visualizations and performance metrics.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.gradcam import GradCAM
from src.eigencam import EigenCAM
from src.config import Config


class CAMComparison:
    """
    Compare GradCAM and EigenCAM methods on the same model and images.
    """
    
    def __init__(self, model_path, config=None):
        """
        Initialize the comparison framework.
        
        Args:
            model_path: Path to the trained model
            config: Configuration object (optional)
        """
        self.config = config or Config()
        self.model = None
        self.gradcam = None
        self.eigencam = None
        self.class_names = []
        
        print(f"Loading model from {model_path}...")
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load the trained model and initialize both CAM methods."""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✓ Model loaded successfully")
            
            # Get class names from config
            if hasattr(self.config, 'DISEASE_CATEGORIES'):
                self.class_names = self.config.DISEASE_CATEGORIES
            else:
                # Default class names
                self.class_names = [
                    'healthy', 'bacterial_infection', 'viral_infection', 
                    'fungal_infection', 'parasitic_infection', 
                    'nutritional_deficiency', 'genetic_disorder'
                ]
            
            # Initialize both CAM methods
            print("\nInitializing GradCAM...")
            self.gradcam = GradCAM(self.model)
            
            print("Initializing EigenCAM...")
            self.eigencam = EigenCAM(self.model)
            
            print("\n✓ Both CAM methods initialized successfully\n")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess an image for the model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=self.config.IMAGE_SIZE
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        return img_array
    
    def compare_on_image(self, image_path, save_dir='comparisons', alpha=0.4):
        """
        Generate and compare both CAM methods on a single image.
        
        Args:
            image_path: Path to the image file
            save_dir: Directory to save comparison results
            alpha: Transparency for heatmap overlay
            
        Returns:
            Dictionary containing comparison results and metrics
        """
        print(f"\n{'='*60}")
        print(f"Processing: {Path(image_path).name}")
        print(f"{'='*60}")
        
        # Load and preprocess image
        image = self.load_and_preprocess_image(image_path)
        
        # Get model prediction
        image_batch = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        print(f"Prediction: {predicted_class} (confidence: {confidence*100:.2f}%)")
        
        # Measure GradCAM performance
        print("\n[GradCAM]")
        start_time = time.time()
        gradcam_result = self.gradcam.generate_heatmap(
            image, 
            class_index=predicted_class_idx, 
            alpha=alpha
        )
        gradcam_time = time.time() - start_time
        print(f"  Generation time: {gradcam_time*1000:.2f}ms")
        print(f"  Target layer: {gradcam_result['target_layer']}")
        
        # Measure EigenCAM performance
        print("\n[EigenCAM]")
        start_time = time.time()
        eigencam_result = self.eigencam.generate_heatmap(
            image, 
            class_index=predicted_class_idx, 
            alpha=alpha
        )
        eigencam_time = time.time() - start_time
        print(f"  Generation time: {eigencam_time*1000:.2f}ms")
        print(f"  Target layer: {eigencam_result['target_layer']}")
        
        # Calculate difference metrics
        heatmap_diff = np.abs(
            gradcam_result['heatmap'] - eigencam_result['heatmap']
        )
        mean_difference = np.mean(heatmap_diff)
        max_difference = np.max(heatmap_diff)
        
        # Calculate spatial correlation
        correlation = np.corrcoef(
            gradcam_result['heatmap'].flatten(),
            eigencam_result['heatmap'].flatten()
        )[0, 1]
        
        print(f"\n[Comparison Metrics]")
        print(f"  Mean heatmap difference: {mean_difference:.4f}")
        print(f"  Max heatmap difference: {max_difference:.4f}")
        print(f"  Spatial correlation: {correlation:.4f}")
        print(f"  Speed ratio (GradCAM/EigenCAM): {gradcam_time/eigencam_time:.2f}x")
        
        # Create visualization
        fig = self._create_comparison_visualization(
            image, 
            gradcam_result, 
            eigencam_result,
            predicted_class,
            confidence
        )
        
        # Save results
        os.makedirs(save_dir, exist_ok=True)
        image_name = Path(image_path).stem
        save_path = os.path.join(save_dir, f"{image_name}_comparison.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison saved to: {save_path}")
        plt.close(fig)
        
        # Compile results
        results = {
            'image_path': image_path,
            'prediction': {
                'class': predicted_class,
                'class_index': int(predicted_class_idx),
                'confidence': confidence
            },
            'gradcam': {
                'time_ms': gradcam_time * 1000,
                'target_layer': gradcam_result['target_layer']
            },
            'eigencam': {
                'time_ms': eigencam_time * 1000,
                'target_layer': eigencam_result['target_layer']
            },
            'metrics': {
                'mean_difference': float(mean_difference),
                'max_difference': float(max_difference),
                'spatial_correlation': float(correlation),
                'speed_ratio': float(gradcam_time / eigencam_time)
            }
        }
        
        return results
    
    def _create_comparison_visualization(self, image, gradcam_result, 
                                        eigencam_result, predicted_class, confidence):
        """
        Create a comprehensive side-by-side comparison visualization.
        
        Args:
            image: Original preprocessed image
            gradcam_result: GradCAM results dictionary
            eigencam_result: EigenCAM results dictionary
            predicted_class: Predicted class name
            confidence: Prediction confidence
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Display image (for visualization)
        display_img = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
        
        # Row 1: Original image and overlays
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(display_img)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gradcam_result['superimposed'])
        ax2.set_title('GradCAM Overlay', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(eigencam_result['superimposed'])
        ax3.set_title('EigenCAM Overlay', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Row 2: Heatmaps only
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(display_img)
        ax4.set_title('Reference', fontsize=11)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        im1 = ax5.imshow(gradcam_result['heatmap'], cmap='jet')
        ax5.set_title('GradCAM Heatmap', fontsize=11)
        ax5.axis('off')
        plt.colorbar(im1, ax=ax5, fraction=0.046, pad=0.04)
        
        ax6 = fig.add_subplot(gs[1, 2])
        im2 = ax6.imshow(eigencam_result['heatmap'], cmap='jet')
        ax6.set_title('EigenCAM Heatmap', fontsize=11)
        ax6.axis('off')
        plt.colorbar(im2, ax=ax6, fraction=0.046, pad=0.04)
        
        # Row 3: Difference and statistics
        ax7 = fig.add_subplot(gs[2, 0:2])
        heatmap_diff = np.abs(gradcam_result['heatmap'] - eigencam_result['heatmap'])
        im3 = ax7.imshow(heatmap_diff, cmap='hot')
        ax7.set_title('Absolute Difference (|GradCAM - EigenCAM|)', fontsize=11)
        ax7.axis('off')
        plt.colorbar(im3, ax=ax7, fraction=0.046, pad=0.04)
        
        # Statistics panel
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        # Calculate metrics
        mean_diff = np.mean(heatmap_diff)
        max_diff = np.max(heatmap_diff)
        correlation = np.corrcoef(
            gradcam_result['heatmap'].flatten(),
            eigencam_result['heatmap'].flatten()
        )[0, 1]
        
        stats_text = f"""
        COMPARISON STATISTICS
        {'─'*30}
        
        Prediction: {predicted_class}
        Confidence: {confidence*100:.2f}%
        
        Mean Difference: {mean_diff:.4f}
        Max Difference: {max_diff:.4f}
        Correlation: {correlation:.4f}
        
        GradCAM Layer:
          {gradcam_result['target_layer'][:25]}...
        
        EigenCAM Layer:
          {eigencam_result['target_layer'][:25]}...
        """
        
        ax8.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        
        # Main title
        fig.suptitle('GradCAM vs EigenCAM Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def batch_comparison(self, image_paths, save_dir='comparisons'):
        """
        Run comparison on multiple images and generate summary statistics.
        
        Args:
            image_paths: List of image file paths
            save_dir: Directory to save results
            
        Returns:
            Dictionary with aggregated results
        """
        all_results = []
        
        print(f"\n{'='*60}")
        print(f"BATCH COMPARISON: {len(image_paths)} images")
        print(f"{'='*60}\n")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\nImage {i}/{len(image_paths)}")
            try:
                result = self.compare_on_image(image_path, save_dir)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Generate summary statistics
        summary = self._generate_summary(all_results)
        
        # Save summary
        summary_path = os.path.join(save_dir, 'comparison_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"\nTotal images processed: {len(all_results)}")
        print(f"\nAverage GradCAM time: {summary['avg_gradcam_time_ms']:.2f}ms")
        print(f"Average EigenCAM time: {summary['avg_eigencam_time_ms']:.2f}ms")
        print(f"Average speed ratio: {summary['avg_speed_ratio']:.2f}x")
        print(f"\nAverage spatial correlation: {summary['avg_correlation']:.4f}")
        print(f"Average mean difference: {summary['avg_mean_difference']:.4f}")
        print(f"\n✓ Summary saved to: {summary_path}\n")
        
        return summary
    
    def _generate_summary(self, results):
        """Generate summary statistics from batch results."""
        if not results:
            return {}
        
        gradcam_times = [r['gradcam']['time_ms'] for r in results]
        eigencam_times = [r['eigencam']['time_ms'] for r in results]
        speed_ratios = [r['metrics']['speed_ratio'] for r in results]
        correlations = [r['metrics']['spatial_correlation'] for r in results]
        mean_diffs = [r['metrics']['mean_difference'] for r in results]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_images': len(results),
            'avg_gradcam_time_ms': np.mean(gradcam_times),
            'std_gradcam_time_ms': np.std(gradcam_times),
            'avg_eigencam_time_ms': np.mean(eigencam_times),
            'std_eigencam_time_ms': np.std(eigencam_times),
            'avg_speed_ratio': np.mean(speed_ratios),
            'avg_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'avg_mean_difference': np.mean(mean_diffs),
            'std_mean_difference': np.std(mean_diffs),
            'detailed_results': results
        }
        
        return summary


def main():
    """Main function to run the comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare GradCAM and EigenCAM explainability methods'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to trained model (.h5 or .keras file)'
    )
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        required=True,
        help='Paths to one or more test images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comparisons',
        help='Output directory for comparison results (default: comparisons)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='Transparency for heatmap overlay (default: 0.4)'
    )
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparison = CAMComparison(args.model)
    
    # Run comparison
    if len(args.images) == 1:
        # Single image comparison
        comparison.compare_on_image(args.images[0], args.output, args.alpha)
    else:
        # Batch comparison
        comparison.batch_comparison(args.images, args.output)


if __name__ == '__main__':
    main()
