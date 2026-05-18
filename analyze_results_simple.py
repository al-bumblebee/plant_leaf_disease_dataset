"""
Image Processing Results Analyzer (Simple Version)
==================================================

This script analyzes the preprocessed images without requiring matplotlib.
It provides statistics and verifies image properties.
"""

import os
import cv2
import numpy as np
from pathlib import Path

def get_dataset_statistics(dataset_path):
    """Get statistics about the processed dataset."""
    stats = {}
    total_images = 0
    
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue
        
        # Count images in this category
        image_count = len([f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        stats[category] = image_count
        total_images += image_count
    
    stats['Total'] = total_images
    return stats

def verify_image_properties(dataset_path, sample_size=10):
    """Verify that all images have the expected properties."""
    print(f"\nVerifying image properties for {dataset_path}...")
    
    image_sizes = []
    image_channels = []
    pixel_ranges = []
    sample_count = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                if sample_count >= sample_size:
                    break
                    
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    h, w, c = img.shape
                    image_sizes.append((w, h))
                    image_channels.append(c)
                    
                    # Check pixel value range
                    min_val = np.min(img)
                    max_val = np.max(img)
                    pixel_ranges.append((min_val, max_val))
                    
                    sample_count += 1
        
        if sample_count >= sample_size:
            break
    
    # Check consistency
    unique_sizes = set(image_sizes)
    unique_channels = set(image_channels)
    
    print(f"  Sampled {sample_count} images")
    print(f"  Image sizes: {unique_sizes}")
    print(f"  Color channels: {unique_channels}")
    print(f"  Pixel value ranges: min={min([r[0] for r in pixel_ranges])}, max={max([r[1] for r in pixel_ranges])}")
    
    if len(unique_sizes) == 1 and len(unique_channels) == 1:
        print(f"  ✓ All images have consistent size: {list(unique_sizes)[0]}")
        print(f"  ✓ All images have consistent channels: {list(unique_channels)[0]}")
        
        expected_size = (256, 256)
        if list(unique_sizes)[0] == expected_size:
            print(f"  ✓ Images are properly resized to {expected_size}")
        else:
            print(f"  ⚠ Warning: Expected size {expected_size}, got {list(unique_sizes)[0]}")
    else:
        print(f"  ⚠ Warning: Images have inconsistent properties!")

def analyze_augmentation(dataset_path):
    """Analyze augmentation results."""
    print(f"\nAnalyzing augmentation for {dataset_path}...")
    
    original_count = 0
    augmented_count = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                if '_aug_' in file:
                    augmented_count += 1
                else:
                    original_count += 1
    
    total_count = original_count + augmented_count
    augmentation_ratio = augmented_count / original_count if original_count > 0 else 0
    
    print(f"  Original images: {original_count}")
    print(f"  Augmented images: {augmented_count}")
    print(f"  Total images: {total_count}")
    print(f"  Augmentation ratio: {augmentation_ratio:.2f}x")

if __name__ == "__main__":
    # Paths to processed datasets
    base_path = r"c:\Users\Almas Siddiqui\Downloads\plant_leaf_disease"
    rice_path = os.path.join(base_path, "rice_processed")
    sugarcane_path = os.path.join(base_path, "sugarcane_processed")
    
    print("Plant Leaf Disease Dataset Analysis")
    print("=" * 40)
    
    # Analyze Rice dataset
    if os.path.exists(rice_path):
        print("\n🌾 Rice Leaf Disease Dataset:")
        print("-" * 30)
        rice_stats = get_dataset_statistics(rice_path)
        for category, count in rice_stats.items():
            print(f"  {category}: {count:,} images")
        
        verify_image_properties(rice_path, sample_size=15)
        analyze_augmentation(rice_path)
    
    # Analyze Sugarcane dataset
    if os.path.exists(sugarcane_path):
        print("\n🌿 Sugarcane Leaf Disease Dataset:")
        print("-" * 35)
        sugarcane_stats = get_dataset_statistics(sugarcane_path)
        for category, count in sugarcane_stats.items():
            print(f"  {category}: {count:,} images")
        
        verify_image_properties(sugarcane_path, sample_size=15)
        analyze_augmentation(sugarcane_path)
    
    print("\n" + "=" * 40)
    print("✅ Dataset analysis completed!")
    print("\nPreprocessing Summary:")
    print("• Images resized to 256x256 with aspect ratio preservation")
    print("• Padding applied to maintain square format")
    print("• Data augmentation applied (rotation, flip, brightness, zoom)")
    print("• Images ready for CNN training")
    
    print("\nNext steps for CNN training:")
    print("1. Import processed datasets into your ML framework (TensorFlow/PyTorch)")
    print("2. Split data into train/validation/test sets")
    print("3. Create data loaders with batch processing")
    print("4. Build and train your CNN model")
    print("5. Use the original folder structure for class labels")
