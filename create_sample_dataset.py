"""
Create a sample dataset with first 50 images from each category
for quick testing and workflow demonstration
"""
import os
import shutil
from pathlib import Path

def create_sample_dataset():
    """Create sample dataset with first 50 images from each category"""
    
    # Source and destination paths
    raw_dir = Path("data/raw")
    sample_dir = Path("data/sample")
    
    # Create sample directory
    sample_dir.mkdir(exist_ok=True)
    
    # Categories to process
    categories = ["healthy", "foot_and_mouth_disease", "lumpy_skin_disease"]
    
    print("🔄 Creating sample dataset with first 50 images from each category...")
    
    for category in categories:
        source_path = raw_dir / category
        dest_path = sample_dir / category
        
        if not source_path.exists():
            print(f"⚠️  {category} folder not found, skipping...")
            continue
            
        # Create destination folder
        dest_path.mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(source_path.glob(f"*{ext}")))
            image_files.extend(list(source_path.glob(f"*{ext.upper()}")))
        
        # Sort to ensure consistent selection
        image_files.sort()
        
        # Copy first 50 images
        copied = 0
        for image_file in image_files[:50]:
            if image_file.name != "README.txt":  # Skip README files
                try:
                    shutil.copy2(image_file, dest_path / image_file.name)
                    copied += 1
                except Exception as e:
                    print(f"Error copying {image_file}: {e}")
        
        print(f"✅ {category}: copied {copied} images")
    
    print(f"\n🎉 Sample dataset created in: {sample_dir.absolute()}")
    print("📊 Dataset summary:")
    
    # Show summary
    total_images = 0
    for category in categories:
        category_path = sample_dir / category
        if category_path.exists():
            count = len([f for f in category_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])
            print(f"   - {category}: {count} images")
            total_images += count
    
    print(f"   - Total: {total_images} images")
    
    return sample_dir

if __name__ == "__main__":
    create_sample_dataset()