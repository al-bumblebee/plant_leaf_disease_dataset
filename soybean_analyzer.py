"""
Soybean Dataset Analyzer
========================

This script analyzes the soybean dataset to determine if data augmentation is needed.
It checks image counts, sizes, quality, and provides recommendations.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoybeanDatasetAnalyzer:
    """Analyzer for soybean plant dataset."""
    
    def __init__(self, dataset_path: str):
        """Initialize the analyzer with dataset path."""
        self.dataset_path = dataset_path
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
    def count_images_by_category(self):
        """Count images in each category."""
        category_counts = {}
        total_images = 0
        
        print("📊 SOYBEAN DATASET ANALYSIS")
        print("=" * 50)
        
        for category in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.isdir(category_path):
                continue
                
            # Count images in this category
            image_files = [f for f in os.listdir(category_path) 
                          if Path(f).suffix in self.supported_extensions]
            count = len(image_files)
            category_counts[category] = count
            total_images += count
            
            print(f"📁 {category}: {count:,} images")
        
        print(f"\n🔢 Total Images: {total_images:,}")
        return category_counts, total_images
    
    def analyze_image_properties(self, sample_size=20):
        """Analyze image properties from sample images."""
        print(f"\n🔍 IMAGE PROPERTIES ANALYSIS (Sample: {sample_size} per category)")
        print("-" * 50)
        
        all_sizes = []
        all_channels = []
        all_file_sizes = []
        category_samples = {}
        
        for category in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.isdir(category_path):
                continue
            
            print(f"\n📂 Analyzing {category}:")
            
            # Get sample images
            image_files = [f for f in os.listdir(category_path) 
                          if Path(f).suffix in self.supported_extensions]
            sample_files = image_files[:sample_size]
            
            category_sizes = []
            category_channels = []
            category_file_sizes = []
            
            for img_file in sample_files:
                img_path = os.path.join(category_path, img_file)
                
                try:
                    # Get file size
                    file_size = os.path.getsize(img_path) / (1024 * 1024)  # MB
                    category_file_sizes.append(file_size)
                    all_file_sizes.append(file_size)
                    
                    # Load image to check properties
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w, c = img.shape
                        category_sizes.append((w, h))
                        category_channels.append(c)
                        all_sizes.append((w, h))
                        all_channels.append(c)
                        
                except Exception as e:
                    logger.warning(f"Could not analyze {img_file}: {e}")
            
            # Analyze this category
            if category_sizes:
                unique_sizes = list(set(category_sizes))
                size_counts = Counter(category_sizes)
                most_common_size = size_counts.most_common(1)[0]
                
                print(f"  • Unique image sizes: {len(unique_sizes)}")
                print(f"  • Most common size: {most_common_size[0]} ({most_common_size[1]} images)")
                print(f"  • File size range: {min(category_file_sizes):.2f} - {max(category_file_sizes):.2f} MB")
                print(f"  • Average file size: {np.mean(category_file_sizes):.2f} MB")
                
                category_samples[category] = {
                    'sizes': category_sizes,
                    'channels': category_channels,
                    'file_sizes': category_file_sizes,
                    'most_common_size': most_common_size[0]
                }
        
        # Overall analysis
        if all_sizes:
            print(f"\n🌍 OVERALL IMAGE PROPERTIES:")
            unique_sizes = list(set(all_sizes))
            size_counts = Counter(all_sizes)
            most_common_size = size_counts.most_common(1)[0]
            
            print(f"  • Total unique sizes: {len(unique_sizes)}")
            print(f"  • Most common size: {most_common_size[0]} ({most_common_size[1]} samples)")
            print(f"  • Color channels: {set(all_channels)}")
            print(f"  • File size range: {min(all_file_sizes):.2f} - {max(all_file_sizes):.2f} MB")
            print(f"  • Average file size: {np.mean(all_file_sizes):.2f} MB")
        
        return category_samples
    
    def assess_class_balance(self, category_counts):
        """Assess class balance and recommend augmentation."""
        print(f"\n⚖️ CLASS BALANCE ANALYSIS")
        print("-" * 50)
        
        if not category_counts:
            print("❌ No categories found!")
            return
        
        total_images = sum(category_counts.values())
        num_classes = len(category_counts)
        ideal_per_class = total_images // num_classes
        
        print(f"📊 Dataset Statistics:")
        print(f"  • Number of classes: {num_classes}")
        print(f"  • Total images: {total_images:,}")
        print(f"  • Ideal images per class: {ideal_per_class:,}")
        
        print(f"\n📈 Class Distribution:")
        max_count = max(category_counts.values())
        min_count = min(category_counts.values())
        
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_images) * 100
            bar_length = int((count / max_count) * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {category:<30} {count:>6,} ({percentage:>5.1f}%) {bar}")
        
        # Calculate imbalance ratio
        imbalance_ratio = max_count / min_count
        print(f"\n📊 Imbalance Analysis:")
        print(f"  • Largest class: {max_count:,} images")
        print(f"  • Smallest class: {min_count:,} images")
        print(f"  • Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        return imbalance_ratio, max_count, min_count
    
    def recommend_augmentation(self, category_counts, imbalance_ratio, max_count, min_count):
        """Provide augmentation recommendations."""
        print(f"\n💡 AUGMENTATION RECOMMENDATIONS")
        print("=" * 50)
        
        total_images = sum(category_counts.values())
        
        # Determine if augmentation is needed
        needs_augmentation = False
        reasons = []
        
        # Check for class imbalance
        if imbalance_ratio > 2.0:
            needs_augmentation = True
            reasons.append(f"Severe class imbalance ({imbalance_ratio:.1f}:1 ratio)")
        elif imbalance_ratio > 1.5:
            needs_augmentation = True
            reasons.append(f"Moderate class imbalance ({imbalance_ratio:.1f}:1 ratio)")
        
        # Check for small dataset size
        if min_count < 500:
            needs_augmentation = True
            reasons.append(f"Small dataset size (minimum {min_count} images per class)")
        
        # Check for total dataset size
        if total_images < 2000:
            needs_augmentation = True
            reasons.append(f"Small total dataset ({total_images:,} images)")
        
        if needs_augmentation:
            print("🚨 AUGMENTATION NEEDED!")
            print("\nReasons:")
            for i, reason in enumerate(reasons, 1):
                print(f"  {i}. {reason}")
            
            print(f"\n📋 Recommended Strategy:")
            
            # Calculate target per class
            target_per_class = max(max_count, 1000)  # At least 1000 or match largest class
            
            print(f"  • Target images per class: {target_per_class:,}")
            
            for category, count in category_counts.items():
                if count < target_per_class:
                    augmentation_factor = target_per_class / count
                    augmentations_needed = target_per_class - count
                    print(f"  • {category}: +{augmentations_needed:,} images ({augmentation_factor:.1f}x factor)")
                else:
                    print(f"  • {category}: No augmentation needed")
            
            print(f"\n🔧 Suggested Augmentation Techniques:")
            print("  • Rotation (±30°)")
            print("  • Horizontal/Vertical flips")
            print("  • Brightness/Contrast adjustment (0.7-1.3)")
            print("  • Zoom/Scale (0.8-1.2)")
            print("  • Gaussian noise (light)")
            print("  • Color jittering (hue/saturation)")
            
            if imbalance_ratio > 3.0:
                print("  • Consider SMOTE or other synthetic data generation")
                print("  • Use weighted loss functions during training")
        
        else:
            print("✅ AUGMENTATION NOT STRICTLY NECESSARY")
            print("\nDataset appears to be:")
            print("  • Reasonably balanced")
            print("  • Sufficient size for training")
            print("\nOptional augmentation could still help:")
            print("  • Improve model generalization")
            print("  • Increase robustness to variations")
            print("  • Boost performance by ~2-5%")
        
        return needs_augmentation, target_per_class if needs_augmentation else None
    
    def generate_augmentation_script_recommendation(self, category_counts, target_per_class):
        """Generate specific script recommendations."""
        print(f"\n🛠️ IMPLEMENTATION RECOMMENDATIONS")
        print("=" * 50)
        
        print("For the Soybean dataset, use the following approach:")
        print("\n1️⃣ Preprocessing Script Configuration:")
        print("```python")
        print("# Recommended settings for Soybean dataset")
        print("preprocessor = SoybeanPreprocessor(")
        print("    target_size=256,")
        print(f"    target_images_per_class={target_per_class}")
        print(")")
        print("```")
        
        print("\n2️⃣ Augmentation Settings:")
        print("```python")
        print("augmentation_settings = {")
        print("    'rotation_range': (-30, 30),")
        print("    'brightness_range': (0.7, 1.3),")
        print("    'contrast_range': (0.7, 1.3),")
        print("    'zoom_range': (0.8, 1.2),")
        print("    'flip_probability': 0.5,")
        print("    'noise_probability': 0.3,")
        print("    'augment_probability': 0.6")
        print("}")
        print("```")
        
        print("\n3️⃣ Expected Results:")
        total_expected = len(category_counts) * target_per_class
        print(f"  • Total images after augmentation: ~{total_expected:,}")
        print(f"  • Balanced classes with {target_per_class:,} images each")
        print("  • Ready for CNN training with improved performance")

def main():
    """Main function to analyze the soybean dataset."""
    # Define dataset path
    dataset_path = r"c:\Users\Almas Siddiqui\Downloads\plant_leaf_disease\soyabean"
    
    # Initialize analyzer
    analyzer = SoybeanDatasetAnalyzer(dataset_path)
    
    try:
        # Step 1: Count images by category
        category_counts, total_images = analyzer.count_images_by_category()
        
        # Step 2: Analyze image properties
        category_samples = analyzer.analyze_image_properties(sample_size=25)
        
        # Step 3: Assess class balance
        imbalance_ratio, max_count, min_count = analyzer.assess_class_balance(category_counts)
        
        # Step 4: Recommend augmentation
        needs_augmentation, target_per_class = analyzer.recommend_augmentation(
            category_counts, imbalance_ratio, max_count, min_count
        )
        
        # Step 5: Generate implementation recommendations
        if needs_augmentation and target_per_class:
            analyzer.generate_augmentation_script_recommendation(category_counts, target_per_class)
        
        print(f"\n🎯 SUMMARY")
        print("=" * 50)
        print(f"Dataset: Soybean Plant Disease")
        print(f"Classes: {len(category_counts)}")
        print(f"Total Images: {total_images:,}")
        print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")
        print(f"Augmentation Needed: {'Yes' if needs_augmentation else 'Optional'}")
        if target_per_class:
            print(f"Recommended Target: {target_per_class:,} images per class")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()
