"""
Blackgram Plant Leaf Disease Image Preprocessor
==============================================

This script preprocesses Blackgram plant leaf disease images for CNN training.
The images are already 512x512 (1:1 aspect ratio), so we just resize to 256x256
without padding and apply data augmentation.

Features:
- Direct resize from 512x512 to 256x256 (no padding needed)
- Data augmentation (rotation, flip, brightness/contrast, zoom)
- Normalization to [0,1]
- Progress tracking and error handling
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Tuple, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlackgramPreprocessor:
    """Blackgram leaf image preprocessor with data augmentation capabilities."""
    
    def __init__(self, target_size: int = 256):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target image size (width and height)
        """
        self.target_size = target_size
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
    def simple_resize(self, image: np.ndarray) -> np.ndarray:
        """
        Simple resize from 512x512 to 256x256 since aspect ratio is already 1:1.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
    
    def apply_rotation(self, image: np.ndarray, angle_range: Tuple[int, int] = (-30, 30)) -> np.ndarray:
        """Apply random rotation to image."""
        angle = random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT_101)
        return rotated
    
    def apply_flip(self, image: np.ndarray) -> np.ndarray:
        """Apply random horizontal/vertical flip."""
        flip_type = random.choice([0, 1, -1])  # vertical, horizontal, both
        return cv2.flip(image, flip_type)
    
    def apply_brightness_contrast(self, image: np.ndarray, 
                                brightness_range: Tuple[float, float] = (0.7, 1.3),
                                contrast_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
        """Apply random brightness and contrast adjustment."""
        brightness = random.uniform(brightness_range[0], brightness_range[1])
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        
        # Convert to float for calculations
        adjusted = image.astype(np.float32)
        adjusted = adjusted * contrast + (brightness - 1) * 255
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)
    
    def apply_zoom(self, image: np.ndarray, zoom_range: Tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
        """Apply random zoom (crop and resize or pad and resize)."""
        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
        h, w = image.shape[:2]
        
        if zoom_factor > 1.0:
            # Zoom in (crop and resize back)
            new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            cropped = image[start_h:start_h+new_h, start_w:start_w+new_w]
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Zoom out (resize and pad)
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Center the resized image with reflection padding
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            zoomed = np.zeros((h, w, 3), dtype=np.uint8)
            zoomed[start_h:start_h+new_h, start_w:start_w+new_w] = resized
            
            # Fill empty areas with reflection of the image
            if start_h > 0:
                zoomed[:start_h, start_w:start_w+new_w] = resized[:start_h, :]
                zoomed[start_h+new_h:, start_w:start_w+new_w] = resized[new_h-start_h:, :]
            if start_w > 0:
                zoomed[:, :start_w] = zoomed[:, start_w:2*start_w]
                zoomed[:, start_w+new_w:] = zoomed[:, start_w+new_w-start_w:start_w+new_w]
        
        return zoomed
    
    def apply_noise(self, image: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
        """Apply random Gaussian noise."""
        if random.random() < 0.3:  # Apply noise with 30% probability
            noise = np.random.normal(0, noise_factor * 255, image.shape).astype(np.float32)
            noisy = image.astype(np.float32) + noise
            noisy = np.clip(noisy, 0, 255)
            return noisy.astype(np.uint8)
        return image
    
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply random blur effect."""
        if random.random() < 0.2:  # Apply blur with 20% probability
            kernel_size = random.choice([3, 5])
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image
    
    def apply_augmentation(self, image: np.ndarray, augment_prob: float = 0.6) -> np.ndarray:
        """
        Apply random data augmentation.
        
        Args:
            image: Input image
            augment_prob: Probability of applying each augmentation
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        # Apply augmentations with given probability
        if random.random() < augment_prob:
            augmented = self.apply_rotation(augmented)
        
        if random.random() < augment_prob:
            augmented = self.apply_flip(augmented)
        
        if random.random() < augment_prob:
            augmented = self.apply_brightness_contrast(augmented)
        
        if random.random() < augment_prob:
            augmented = self.apply_zoom(augmented)
        
        # Apply additional augmentations
        augmented = self.apply_noise(augmented)
        augmented = self.apply_blur(augmented)
        
        return augmented
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1]."""
        return image.astype(np.float32) / 255.0
    
    def count_images(self, input_dir: str) -> int:
        """Count total number of images in dataset."""
        total = 0
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if Path(file).suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    total += 1
        return total
    
    def preprocess_dataset(self, input_dir: str, output_dir: str, 
                         apply_augmentation: bool = True, 
                         augmentation_factor: int = 2,
                         augment_prob: float = 0.6) -> None:
        """
        Preprocess entire Blackgram dataset.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            apply_augmentation: Whether to apply data augmentation
            augmentation_factor: Number of augmented versions per image
            augment_prob: Probability of applying each augmentation technique
        """
        logger.info(f"Starting preprocessing of Blackgram dataset: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Target size: {self.target_size}x{self.target_size}")
        logger.info(f"Apply augmentation: {apply_augmentation}")
        
        if apply_augmentation:
            logger.info(f"Augmentation factor: {augmentation_factor}")
            logger.info(f"Augmentation probability: {augment_prob}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Count total images for progress bar
        total_images = self.count_images(input_dir)
        logger.info(f"Found {total_images} images to process")
        
        processed_count = 0
        error_count = 0
        
        # Progress bar
        with tqdm(total=total_images, desc="Processing Blackgram images") as pbar:
            # Walk through all subdirectories
            for root, dirs, files in os.walk(input_dir):
                # Get relative path from input_dir
                rel_path = os.path.relpath(root, input_dir)
                
                # Create corresponding output directory
                if rel_path == '.':
                    output_subdir = output_dir
                else:
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                
                # Process each image file
                for file in files:
                    file_path = Path(file)
                    if file_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                        continue
                    
                    input_path = os.path.join(root, file)
                    
                    try:
                        # Load image
                        image = cv2.imread(input_path)
                        if image is None:
                            logger.warning(f"Could not load image: {input_path}")
                            error_count += 1
                            pbar.update(1)
                            continue
                        
                        # Simple resize (no padding needed for 512x512 -> 256x256)
                        processed_image = self.simple_resize(image)
                        
                        # Save original processed image
                        output_path = os.path.join(output_subdir, file)
                        cv2.imwrite(output_path, processed_image)
                        processed_count += 1
                        
                        # Apply augmentation if requested
                        if apply_augmentation:
                            for aug_idx in range(augmentation_factor):
                                # Apply augmentation
                                augmented = self.apply_augmentation(processed_image, augment_prob)
                                
                                # Save augmented image
                                name_parts = file_path.stem, aug_idx + 1, file_path.suffix
                                aug_filename = f"{name_parts[0]}_aug_{name_parts[1]}{name_parts[2]}"
                                aug_output_path = os.path.join(output_subdir, aug_filename)
                                
                                cv2.imwrite(aug_output_path, augmented)
                                processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing {input_path}: {str(e)}")
                        error_count += 1
                    
                    pbar.update(1)
        
        logger.info(f"Blackgram preprocessing completed!")
        logger.info(f"Successfully processed: {processed_count} images")
        logger.info(f"Errors encountered: {error_count} images")

def main():
    """Main function to run the Blackgram preprocessor."""
    print("Blackgram Plant Leaf Disease Image Preprocessor")
    print("=" * 50)
    
    # Define paths
    base_path = r"c:\Users\Almas Siddiqui\Downloads\plant_leaf_disease"
    blackgram_input = os.path.join(base_path, "Blackgram Plant Leaf Disease Dataset")
    blackgram_output = os.path.join(base_path, "blackgram_processed")
    
    # Initialize preprocessor
    preprocessor = BlackgramPreprocessor(target_size=256)
    
    print("\nProcessing Blackgram Plant Leaf Disease Dataset...")
    if os.path.exists(blackgram_input):
        preprocessor.preprocess_dataset(
            input_dir=blackgram_input,
            output_dir=blackgram_output,
            apply_augmentation=True,  # Enable augmentation
            augmentation_factor=2,    # Create 2 augmented versions per image
            augment_prob=0.6         # 60% chance for each augmentation
        )
        
        print(f"\n✅ Processing completed!")
        print(f"Check output directory: {blackgram_output}")
        
        # Show basic statistics
        print("\n📊 Dataset Statistics:")
        if os.path.exists(blackgram_output):
            total_processed = 0
            for category in os.listdir(blackgram_output):
                category_path = os.path.join(blackgram_output, category)
                if os.path.isdir(category_path):
                    count = len([f for f in os.listdir(category_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  {category}: {count:,} images")
                    total_processed += count
            print(f"  Total: {total_processed:,} images")
    else:
        print(f"❌ Blackgram dataset not found at: {blackgram_input}")
        print("Please make sure the dataset is in the correct location.")

if __name__ == "__main__":
    main()
