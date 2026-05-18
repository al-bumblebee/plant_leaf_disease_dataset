"""
Plant Leaf Disease Image Preprocessor
=====================================

This script preprocesses plant leaf images (Rice and Sugarcane) for CNN training.
Features:
- Resize with padding to maintain aspect ratio
- Data augmentation (rotation, flip, brightness/contrast, zoom)
- Normalization to [0,1]
- Support for both JPEG and PNG files
- Progress tracking and error handling
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from typing import Tuple, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlantLeafPreprocessor:
    """Plant leaf image preprocessor with data augmentation capabilities."""
    
    def __init__(self, target_size: int = 256, padding_color: Tuple[int, int, int] = (128, 128, 128)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target image size (width and height)
            padding_color: RGB color for padding (default: gray)
        """
        self.target_size = target_size
        self.padding_color = padding_color
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
    def resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio using padding.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Resized image with padding
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calculate padding
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        # Add padding
        padded_image = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=self.padding_color
        )
        
        return padded_image
    
    def apply_rotation(self, image: np.ndarray, angle_range: Tuple[int, int] = (-30, 30)) -> np.ndarray:
        """Apply random rotation to image."""
        angle = random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=self.padding_color)
        return rotated
    
    def apply_flip(self, image: np.ndarray) -> np.ndarray:
        """Apply random horizontal/vertical flip."""
        flip_type = random.choice([0, 1, -1])  # vertical, horizontal, both
        return cv2.flip(image, flip_type)
    
    def apply_brightness_contrast(self, image: np.ndarray, 
                                brightness_range: Tuple[float, float] = (0.8, 1.2),
                                contrast_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply random brightness and contrast adjustment."""
        brightness = random.uniform(brightness_range[0], brightness_range[1])
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        
        # Convert to float for calculations
        adjusted = image.astype(np.float32)
        adjusted = adjusted * contrast + (brightness - 1) * 255
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)
    
    def apply_zoom(self, image: np.ndarray, zoom_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Apply random zoom (crop and resize)."""
        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
        h, w = image.shape[:2]
        
        if zoom_factor > 1.0:
            # Zoom in (crop)
            new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            cropped = image[start_h:start_h+new_h, start_w:start_w+new_w]
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Zoom out (resize and pad)
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Center the resized image
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            zoomed = np.full((h, w, 3), self.padding_color, dtype=np.uint8)
            zoomed[start_h:start_h+new_h, start_w:start_w+new_w] = resized
        
        return zoomed
    
    def apply_augmentation(self, image: np.ndarray, augment_prob: float = 0.5) -> np.ndarray:
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
                         apply_augmentation: bool = False, 
                         augmentation_factor: int = 2,
                         augment_prob: float = 0.5) -> None:
        """
        Preprocess entire dataset.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            apply_augmentation: Whether to apply data augmentation
            augmentation_factor: Number of augmented versions per image
            augment_prob: Probability of applying each augmentation technique
        """
        logger.info(f"Starting preprocessing of {input_dir}")
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
        with tqdm(total=total_images, desc="Processing images") as pbar:
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
                        
                        # Resize with padding
                        processed_image = self.resize_with_padding(image)
                        
                        # Save original processed image
                        output_path = os.path.join(output_subdir, file)
                        # Convert back to uint8 for saving
                        save_image = processed_image.astype(np.uint8)
                        cv2.imwrite(output_path, save_image)
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
                                
                                save_image = augmented.astype(np.uint8)
                                cv2.imwrite(aug_output_path, save_image)
                                processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing {input_path}: {str(e)}")
                        error_count += 1
                    
                    pbar.update(1)
        
        logger.info(f"Preprocessing completed!")
        logger.info(f"Successfully processed: {processed_count} images")
        logger.info(f"Errors encountered: {error_count} images")

def main():
    """Main function to run the preprocessor."""
    parser = argparse.ArgumentParser(description="Plant Leaf Disease Image Preprocessor")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing the dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed images")
    parser.add_argument("--target_size", type=int, default=256,
                       help="Target image size (default: 256)")
    parser.add_argument("--padding_color", type=int, nargs=3, default=[128, 128, 128],
                       help="RGB padding color (default: 128 128 128)")
    parser.add_argument("--augment", action="store_true",
                       help="Apply data augmentation")
    parser.add_argument("--aug_factor", type=int, default=2,
                       help="Number of augmented versions per image (default: 2)")
    parser.add_argument("--aug_prob", type=float, default=0.5,
                       help="Probability of applying each augmentation (default: 0.5)")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = PlantLeafPreprocessor(
        target_size=args.target_size,
        padding_color=tuple(args.padding_color)
    )
    
    # Process dataset
    preprocessor.preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        apply_augmentation=args.augment,
        augmentation_factor=args.aug_factor,
        augment_prob=args.aug_prob
    )

if __name__ == "__main__":
    # Example usage for your specific datasets
    print("Plant Leaf Disease Image Preprocessor")
    print("=" * 40)
    
    # Define paths based on your workspace structure
    base_path = r"c:\Users\Almas Siddiqui\Downloads\plant_leaf_disease"
    rice_input = os.path.join(base_path, "rice_leaf_diseases")
    sugarcane_input = os.path.join(base_path, "Sugarcane Leaf Disease Dataset")
    
    # Create output directories
    rice_output = os.path.join(base_path, "rice_processed")
    sugarcane_output = os.path.join(base_path, "sugarcane_processed")
    
    # Initialize preprocessor
    preprocessor = PlantLeafPreprocessor(target_size=256, padding_color=(128, 128, 128))
    
    print("\nProcessing Rice Leaf Disease Dataset...")
    if os.path.exists(rice_input):
        preprocessor.preprocess_dataset(
            input_dir=rice_input,
            output_dir=rice_output,
            apply_augmentation=True,  # Enable augmentation
            augmentation_factor=2,    # Create 2 augmented versions per image
            augment_prob=0.5         # 50% chance for each augmentation
        )
    else:
        print(f"Rice dataset not found at: {rice_input}")
    
    print("\nProcessing Sugarcane Leaf Disease Dataset...")
    if os.path.exists(sugarcane_input):
        preprocessor.preprocess_dataset(
            input_dir=sugarcane_input,
            output_dir=sugarcane_output,
            apply_augmentation=True,  # Enable augmentation
            augmentation_factor=2,    # Create 2 augmented versions per image
            augment_prob=0.5         # 50% chance for each augmentation
        )
    else:
        print(f"Sugarcane dataset not found at: {sugarcane_input}")
    
    print("\nPreprocessing completed! Check the output directories:")
    print(f"Rice processed: {rice_output}")
    print(f"Sugarcane processed: {sugarcane_output}")
