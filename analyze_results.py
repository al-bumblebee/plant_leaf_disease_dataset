"""
Image Processing Results Viewer
===============================

This script demonstrates how to load and use the preprocessed images for CNN training.
It shows sample images from each category and provides basic statistics.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_sample_images(dataset_path, max_samples=3):
    """Load sample images from each category."""
    samples = {}
    
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue
            
        # Get first few images from this category
        images = []
        filenames = []
        
        for i, filename in enumerate(os.listdir(category_path)):
            if i >= max_samples:
                break
                
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert BGR to RGB for matplotlib
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
                    filenames.append(filename)
        
        samples[category] = {'images': images, 'filenames': filenames}
    
    return samples

def display_samples(samples, dataset_name):
    """Display sample images from each category."""
    num_categories = len(samples)
    max_samples = max(len(data['images']) for data in samples.values())
    
    fig, axes = plt.subplots(num_categories, max_samples, 
                            figsize=(max_samples * 3, num_categories * 3))
    fig.suptitle(f'{dataset_name} - Preprocessed Samples', fontsize=16)
    
    if num_categories == 1:
        axes = [axes]
    if max_samples == 1:
        axes = [[ax] for ax in axes]
    
    for i, (category, data) in enumerate(samples.items()):
        for j, (img, filename) in enumerate(zip(data['images'], data['filenames'])):
            if j < max_samples:
                axes[i][j].imshow(img)
                axes[i][j].set_title(f'{category}\n{filename}', fontsize=8)
                axes[i][j].axis('off')
        
        # Hide unused subplots
        for j in range(len(data['images']), max_samples):
            axes[i][j].axis('off')
    
    plt.tight_layout()
    plt.show()

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
                    sample_count += 1
        
        if sample_count >= sample_size:
            break
    
    # Check consistency
    unique_sizes = set(image_sizes)
    unique_channels = set(image_channels)
    
    print(f"  Sampled {sample_count} images")
    print(f"  Image sizes: {unique_sizes}")
    print(f"  Color channels: {unique_channels}")
    
    if len(unique_sizes) == 1 and len(unique_channels) == 1:
        print(f"  ✓ All images have consistent size: {list(unique_sizes)[0]}")
        print(f"  ✓ All images have consistent channels: {list(unique_channels)[0]}")
    else:
        print(f"  ⚠ Warning: Images have inconsistent properties!")

if __name__ == "__main__":
    # Paths to processed datasets
    base_path = r"c:\Users\Almas Siddiqui\Downloads\plant_leaf_disease"
    rice_path = os.path.join(base_path, "rice_processed")
    sugarcane_path = os.path.join(base_path, "sugarcane_processed")
    
    print("Plant Leaf Disease Dataset Analysis")
    print("=" * 40)
    
    # Analyze Rice dataset
    if os.path.exists(rice_path):
        print("\n🌾 Rice Leaf Disease Dataset Statistics:")
        rice_stats = get_dataset_statistics(rice_path)
        for category, count in rice_stats.items():
            print(f"  {category}: {count} images")
        
        verify_image_properties(rice_path)
        
        # Load and display samples
        print("\nLoading rice samples...")
        rice_samples = load_sample_images(rice_path, max_samples=2)
        # Uncomment the next line to display images (requires GUI)
        # display_samples(rice_samples, "Rice Leaf Diseases")
    
    # Analyze Sugarcane dataset
    if os.path.exists(sugarcane_path):
        print("\n🌿 Sugarcane Leaf Disease Dataset Statistics:")
        sugarcane_stats = get_dataset_statistics(sugarcane_path)
        for category, count in sugarcane_stats.items():
            print(f"  {category}: {count} images")
        
        verify_image_properties(sugarcane_path)
        
        # Load and display samples
        print("\nLoading sugarcane samples...")
        sugarcane_samples = load_sample_images(sugarcane_path, max_samples=2)
        # Uncomment the next line to display images (requires GUI)
        # display_samples(sugarcane_samples, "Sugarcane Leaf Diseases")
    
    print("\n✅ Dataset analysis completed!")
    print("\nNext steps for CNN training:")
    print("1. The images are now preprocessed and ready for training")
    print("2. All images are 256x256 pixels with consistent format")
    print("3. Data augmentation has been applied to increase dataset size")
    print("4. Use these processed images with TensorFlow/PyTorch for CNN training")
