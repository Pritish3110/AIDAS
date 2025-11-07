#!/usr/bin/env python3
"""
Test script to verify dataset distribution visualization
"""
import json
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

from enhanced_app import create_dataset_distribution_plot

def test_dataset_plot():
    """Test the dataset distribution plot generation"""
    
    # Load the actual results data
    try:
        with open('models/enhanced_90plus_results.json', 'r') as f:
            results_data = json.load(f)
        
        print("📊 Testing Dataset Distribution Plot")
        print("=" * 50)
        
        # Print the dataset summary from the file
        dataset_summary = results_data.get('dataset_summary', {})
        print("\n📋 Dataset Summary from JSON:")
        for class_name, data in dataset_summary.items():
            print(f"  {class_name}:")
            print(f"    Train: {data['train']}")
            print(f"    Validation: {data['validation']}")  
            print(f"    Test: {data['test']}")
            print(f"    Total Available: {data['total_available']}")
        
        # Calculate totals manually
        total_train = sum([v['train'] for v in dataset_summary.values()])
        total_val = sum([v['validation'] for v in dataset_summary.values()])
        total_test = sum([v['test'] for v in dataset_summary.values()])
        total_all = total_train + total_val + total_test
        total_available = sum([v['total_available'] for v in dataset_summary.values()])
        
        print(f"\n📈 Calculated Totals:")
        print(f"  Total Training: {total_train}")
        print(f"  Total Validation: {total_val}")
        print(f"  Total Test: {total_test}")
        print(f"  Total Used: {total_all}")
        print(f"  Total Available: {total_available}")
        print(f"  Utilization: {(total_all/total_available*100):.1f}%")
        
        # Test plot generation
        try:
            print(f"\n🎨 Testing plot generation...")
            base64_image = create_dataset_distribution_plot(results_data)
            print(f"✅ Dataset distribution plot generated successfully!")
            print(f"📏 Image data length: {len(base64_image)} characters")
            
            # Save a sample HTML to test the image
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Dataset Distribution Test</title></head>
            <body>
                <h2>Dataset Distribution Test</h2>
                <img src="{base64_image}" style="max-width: 100%; height: auto;">
            </body>
            </html>
            """
            
            with open('test_dataset_plot.html', 'w') as f:
                f.write(html_content)
            
            print(f"📄 Test HTML saved as: test_dataset_plot.html")
            print(f"🌐 Open this file in a browser to verify the plot")
            
        except Exception as e:
            print(f"❌ Error generating plot: {e}")
            return False
            
        return True
        
    except FileNotFoundError:
        print("❌ Results file not found: models/enhanced_90plus_results.json")
        return False
    except Exception as e:
        print(f"❌ Error loading results data: {e}")
        return False

if __name__ == "__main__":
    success = test_dataset_plot()
    if success:
        print(f"\n🎉 Dataset plot test completed successfully!")
    else:
        print(f"\n💥 Dataset plot test failed!")