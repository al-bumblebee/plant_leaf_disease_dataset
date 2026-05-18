"""
Enhanced Blackgram Plant Leaf Disease Image Preprocessor
======================================================

This script creates approximately 1000 images per class using advanced augmentation.
Optimized for balanced dataset creation with realistic augmentations.
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Tuple, List, Optional
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBlackgramPreprocessor:
    """Enhanced Blackgram preprocessor for generating ~1000 images per class."""
    
    def __init__(self, target_size: int = 256, target_images_per_class: int = 1000):
        """
        Initialize the enhanced preprocessor.
        
        Args:
            target_size: Target image size (width and height)
            target_images_per_class: Target number of images per class
        """
        self.target_size = target_size
        self.target_images_per_class = target_images_per_class
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
    def simple_resize(self, image: np.ndarray) -> np.ndarray:
        """Simple resize from 512x512 to 256x256."""
        return cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
    
    def apply_rotation(self, image: np.ndarray, angle_range: Tuple[int, int] = (-45, 45)) -> np.ndarray:
        """Apply random rotation (enhanced range)."""
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
                                brightness_range: Tuple[float, float] = (0.6, 1.4),
                                contrast_range: Tuple[float, float] = (0.6, 1.4)) -> np.ndarray:
        """Apply random brightness and contrast (enhanced range)."""
        brightness = random.uniform(brightness_range[0], brightness_range[1])
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        
        adjusted = image.astype(np.float32)
        adjusted = adjusted * contrast + (brightness - 1) * 255
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)
    
    def apply_zoom(self, image: np.ndarray, zoom_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply random zoom (enhanced range)."""
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
            
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            zoomed = np.zeros((h, w, 3), dtype=np.uint8)
            zoomed[start_h:start_h+new_h, start_w:start_w+new_w] = resized
            
            # Fill with reflection
            if start_h > 0:
                zoomed[:start_h, start_w:start_w+new_w] = resized[:start_h, :]
                zoomed[start_h+new_h:, start_w:start_w+new_w] = resized[new_h-start_h:, :]
            if start_w > 0:
                zoomed[:, :start_w] = zoomed[:, start_w:2*start_w]
                zoomed[:, start_w+new_w:] = zoomed[:, start_w+new_w-start_w:start_w+new_w]
        
        return zoomed
    
    def apply_noise(self, image: np.ndarray, noise_factor: float = 0.08) -> np.ndarray:
        """Apply random Gaussian noise (enhanced)."""
        if random.random() < 0.4:  # 40% probability
            noise = np.random.normal(0, noise_factor * 255, image.shape).astype(np.float32)
            noisy = image.astype(np.float32) + noise
            noisy = np.clip(noisy, 0, 255)
            return noisy.astype(np.uint8)
        return image
    
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply random blur effect (enhanced)."""
        if random.random() < 0.25:  # 25% probability
            kernel_size = random.choice([3, 5, 7])
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image
    
    def apply_shear(self, image: np.ndarray, shear_range: float = 0.2) -> np.ndarray:
        """Apply random shear transformation."""
        if random.random() < 0.3:  # 30% probability
            shear_x = random.uniform(-shear_range, shear_range)
            shear_y = random.uniform(-shear_range, shear_range)
            
            h, w = image.shape[:2]
            shear_matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
            
            sheared = cv2.warpAffine(image, shear_matrix, (w, h), 
                                   borderMode=cv2.BORDER_REFLECT_101)
            return sheared
        return image
    
    def apply_hue_saturation(self, image: np.ndarray) -> np.ndarray:
        """Apply random hue and saturation changes."""
        if random.random() < 0.4:  # 40% probability
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Hue shift
            hue_shift = random.uniform(-20, 20)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            # Saturation adjustment
            sat_factor = random.uniform(0.7, 1.3)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
            
            modified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            return modified
        return image
    
    def apply_shadow(self, image: np.ndarray) -> np.ndarray:
        """Apply random shadow effect."""
        if random.random() < 0.2:  # 20% probability
            h, w = image.shape[:2]
            
            # Create random shadow mask
            shadow = np.zeros((h, w), dtype=np.float32)
            
            # Random shadow parameters
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            radius = random.randint(h//4, h//2)
            
            y, x = np.ogrid[:h, :w]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            shadow[mask] = random.uniform(0.3, 0.7)
            shadow = cv2.GaussianBlur(shadow, (51, 51), 0)
            
            # Apply shadow
            shadow_factor = 1 - shadow[:, :, np.newaxis]
            shadowed = (image.astype(np.float32) * shadow_factor).astype(np.uint8)
            return shadowed
        return image
    
    def create_augmentation_variants(self, image: np.ndarray, num_variants: int) -> List[np.ndarray]:
        """Create multiple augmentation variants of an image."""
        variants = []
        
        for i in range(num_variants):
            augmented = image.copy()
            
            # Randomly select and apply augmentations
            augmentation_count = random.randint(1, 4)  # Apply 1-4 augmentations
            
            available_augmentations = [
                lambda img: self.apply_rotation(img),
                lambda img: self.apply_flip(img),
                lambda img: self.apply_brightness_contrast(img),
                lambda img: self.apply_zoom(img),
                lambda img: self.apply_shear(img),
                lambda img: self.apply_hue_saturation(img),
                lambda img: self.apply_shadow(img),
                lambda img: self.apply_noise(img),
                lambda img: self.apply_blur(img)
            ]
            
            # Randomly select augmentations
            selected_augmentations = random.sample(available_augmentations, 
                                                 min(augmentation_count, len(available_augmentations)))
            
            # Apply selected augmentations
            for aug_func in selected_augmentations:
                try:
                    augmented = aug_func(augmented)
                except:
                    continue  # Skip if augmentation fails
            
            variants.append(augmented)
        
        return variants
    
    def preprocess_dataset_balanced(self, input_dir: str, output_dir: str) -> None:
        """
        Preprocess dataset to create balanced classes with target number of images.
        """
        logger.info(f"Starting balanced preprocessing of Blackgram dataset: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Target size: {self.target_size}x{self.target_size}")
        logger.info(f"Target images per class: {self.target_images_per_class}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each class directory
        class_stats = {}
        
        for class_dir in os.listdir(input_dir):
            class_path = os.path.join(input_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            # Count original images
            original_images = [f for f in os.listdir(class_path) 
                             if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            original_count = len(original_images)
            
            if original_count == 0:
                logger.warning(f"No images found in {class_dir}")
                continue
            
            # Calculate how many augmentations needed
            augmentations_needed = max(0, self.target_images_per_class - original_count)
            augmentations_per_image = math.ceil(augmentations_needed / original_count)
            
            logger.info(f"\nProcessing class: {class_dir}")
            logger.info(f"Original images: {original_count}")
            logger.info(f"Target images: {self.target_images_per_class}")
            logger.info(f"Augmentations per image: {augmentations_per_image}")
            
            # Create output class directory
            output_class_dir = os.path.join(output_dir, class_dir)
            os.makedirs(output_class_dir, exist_ok=True)
            
            processed_count = 0
            
            # Process each image with progress bar
            with tqdm(total=len(original_images), desc=f"Processing {class_dir}") as pbar:
                for img_file in original_images:
                    img_path = os.path.join(class_path, img_file)
                    
                    try:
                        # Load and resize original image
                        image = cv2.imread(img_path)
                        if image is None:
                            continue
                        
                        processed_image = self.simple_resize(image)
                        
                        # Save original processed image
                        output_path = os.path.join(output_class_dir, img_file)
                        cv2.imwrite(output_path, processed_image)
                        processed_count += 1
                        
                        # Create augmentation variants
                        if augmentations_per_image > 0:
                            variants = self.create_augmentation_variants(
                                processed_image, augmentations_per_image
                            )
                            
                            for idx, variant in enumerate(variants):
                                if processed_count >= self.target_images_per_class:
                                    break
                                
                                file_stem = Path(img_file).stem
                                file_ext = Path(img_file).suffix
                                aug_filename = f"{file_stem}_aug_{idx+1}{file_ext}"
                                aug_path = os.path.join(output_class_dir, aug_filename)
                                
                                cv2.imwrite(aug_path, variant)
                                processed_count += 1
                        
                        if processed_count >= self.target_images_per_class:
                            break
                            
                    except Exception as e:
                        logger.error(f"Error processing {img_file}: {str(e)}")
                    
                    pbar.update(1)
            
            class_stats[class_dir] = {
                'original': original_count,
                'processed': processed_count,
                'augmentation_ratio': processed_count / original_count if original_count > 0 else 0
            }
        
        # Print final statistics
        logger.info(f"\n🎉 Balanced preprocessing completed!")
        print("\n📊 Final Dataset Statistics:")
        print("="*60)
        
        total_original = 0
        total_processed = 0
        
        for class_name, stats in class_stats.items():
            print(f"{class_name}:")
            print(f"  Original: {stats['original']:,} images")
            print(f"  Final: {stats['processed']:,} images")
            print(f"  Augmentation ratio: {stats['augmentation_ratio']:.2f}x")
            print()
            
            total_original += stats['original']
            total_processed += stats['processed']
        
        print(f"TOTAL:")
        print(f"  Original: {total_original:,} images")
        print(f"  Final: {total_processed:,} images")
        print(f"  Overall ratio: {total_processed/total_original:.2f}x")


def print_augmentation_comparison():
    """Print comparison between Blackgram and Plant Leaf preprocessors."""
    print("\n🔄 AUGMENTATION TECHNIQUES COMPARISON")
    print("="*60)

    print("\n1️⃣ ENHANCED BLACKGRAM PREPROCESSOR:")
    print("-" * 30)
    print("✅ Rotation: ±45° (enhanced from ±30°)")
    print("✅ Flip: Horizontal/Vertical/Both")  
    print("✅ Brightness: 0.6-1.4 (enhanced from 0.7-1.3)")
    print("✅ Contrast: 0.6-1.4 (enhanced from 0.7-1.3)")
    print("✅ Zoom: 0.8-1.2 (enhanced from 0.85-1.15)")
    print("✅ Gaussian Noise: 8% factor, 40% probability")
    print("✅ Gaussian Blur: 25% probability, kernels 3/5/7")
    print("🆕 Shear Transform: ±20%, 30% probability")
    print("🆕 Hue/Saturation: ±20 hue, 0.7-1.3 sat, 40% probability")
    print("🆕 Shadow Effects: Random circular shadows, 20% probability")
    print("🔧 Multi-augmentation: 1-4 techniques per image")

    print("\n2️⃣ PLANT LEAF PREPROCESSOR:")
    print("-" * 30)
    print("✅ Rotation: ±30°")
    print("✅ Flip: Horizontal/Vertical/Both")
    print("✅ Brightness: 0.8-1.2")
    print("✅ Contrast: 0.8-1.2") 
    print("✅ Zoom: 0.9-1.1")
    print("🔧 Resize with Padding: For aspect ratio preservation")
    print("🔧 Border handling: Constant color padding")

    print("\n3️⃣ KEY DIFFERENCES:")
    print("-" * 30)
    print("🎯 Purpose:")
    print("  • Enhanced Blackgram: Balanced dataset creation (~1000/class)")
    print("  • Plant Leaf: Fixed multiplier with aspect preservation")

    print("\n🔧 Augmentation Intensity:")
    print("  • Enhanced Blackgram: Aggressive, multi-technique approach")
    print("  • Plant Leaf: Conservative, single-technique approach")

    print("\n🆕 Unique Features:")
    print("  • Enhanced Blackgram: 9 techniques, adaptive generation")
    print("  • Plant Leaf: Aspect ratio preservation with padding")


def main():
    """Main function to run the enhanced Blackgram preprocessor."""
    print("Enhanced Blackgram Plant Leaf Disease Image Preprocessor")
    print("=" * 60)
    print("🎯 Target: ~1000 images per class with advanced augmentation")
    
    # Define paths
    base_path = r"c:\Users\Almas Siddiqui\Downloads\plant_leaf_disease"
    blackgram_input = os.path.join(base_path, "Blackgram Plant Leaf Disease Dataset")
    blackgram_output = os.path.join(base_path, "blackgram_balanced_1000")
    
    # Initialize enhanced preprocessor
    preprocessor = EnhancedBlackgramPreprocessor(
        target_size=256, 
        target_images_per_class=1000
    )
    
    print("\nProcessing Blackgram Dataset for Balanced Classes...")
    if os.path.exists(blackgram_input):
        preprocessor.preprocess_dataset_balanced(
            input_dir=blackgram_input,
            output_dir=blackgram_output
        )
        
        print(f"\n✅ Enhanced processing completed!")
        print(f"📁 Check output directory: {blackgram_output}")
        
        # Print comparison
        print_augmentation_comparison()
        
    else:
        print(f"❌ Blackgram dataset not found at: {blackgram_input}")

if __name__ == "__main__":
    main()
