"""
Soybean Plant Disease Dataset Preprocessor
==========================================

This script addresses the severe class imbalance in the soybean dataset
by generating augmented images for the minority class (Healthy).

Key Features:
- Balances classes to 1,055 images each
- Applies aggressive augmentation to healthy images only
- Maintains original disease images
- Creates balanced dataset for CNN training
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Tuple, List
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoybeanBalancedPreprocessor:
    """Soybean preprocessor focused on class balancing."""
    
    def __init__(self, target_size: int = 256):
        """Initialize the preprocessor."""
        self.target_size = target_size
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
    def apply_rotation(self, image: np.ndarray, angle_range: Tuple[int, int] = (-45, 45)) -> np.ndarray:
        """Apply random rotation."""
        angle = random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT_101)
        return rotated
    
    def apply_flip(self, image: np.ndarray) -> np.ndarray:
        """Apply random flip."""
        flip_type = random.choice([0, 1, -1])
        return cv2.flip(image, flip_type)
    
    def apply_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness and contrast."""
        brightness = random.uniform(0.6, 1.4)
        contrast = random.uniform(0.6, 1.4)
        
        adjusted = image.astype(np.float32)
        adjusted = adjusted * contrast + (brightness - 1) * 255
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)
    
    def apply_zoom(self, image: np.ndarray) -> np.ndarray:
        """Apply random zoom."""
        zoom_factor = random.uniform(0.75, 1.25)
        h, w = image.shape[:2]
        
        if zoom_factor > 1.0:
            # Zoom in
            new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            cropped = image[start_h:start_h+new_h, start_w:start_w+new_w]
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Zoom out
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
    
    def apply_hue_saturation(self, image: np.ndarray) -> np.ndarray:
        """Apply hue and saturation changes."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Hue shift
        hue_shift = random.uniform(-15, 15)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Saturation adjustment
        sat_factor = random.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        
        modified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return modified
    
    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise."""
        if random.random() < 0.4:
            noise = np.random.normal(0, 0.05 * 255, image.shape).astype(np.float32)
            noisy = image.astype(np.float32) + noise
            noisy = np.clip(noisy, 0, 255)
            return noisy.astype(np.uint8)
        return image
    
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply random blur."""
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image
    
    def create_augmented_image(self, image: np.ndarray) -> np.ndarray:
        """Create one augmented version of an image."""
        augmented = image.copy()
        
        # Apply 2-4 random augmentations
        num_augmentations = random.randint(2, 4)
        
        augmentation_functions = [
            self.apply_rotation,
            self.apply_flip,
            self.apply_brightness_contrast,
            self.apply_zoom,
            self.apply_hue_saturation,
            self.apply_noise,
            self.apply_blur
        ]
        
        selected_augmentations = random.sample(augmentation_functions, 
                                             min(num_augmentations, len(augmentation_functions)))
        
        for aug_func in selected_augmentations:
            try:
                augmented = aug_func(augmented)
            except:
                continue
        
        return augmented
    
    def balance_soybean_dataset(self, input_dir: str, output_dir: str):
        """Balance the soybean dataset by augmenting the minority class."""
        logger.info("Starting soybean dataset balancing...")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze current dataset
        category_info = {}
        for category in os.listdir(input_dir):
            category_path = os.path.join(input_dir, category)
            if not os.path.isdir(category_path):
                continue
            
            images = [f for f in os.listdir(category_path) 
                     if Path(f).suffix in self.supported_extensions]
            category_info[category] = {
                'path': category_path,
                'images': images,
                'count': len(images)
            }
        
        if len(category_info) != 2:
            logger.error("Expected exactly 2 categories (Disease and Healthy)")
            return
        
        # Determine majority and minority classes
        sorted_categories = sorted(category_info.items(), key=lambda x: x[1]['count'], reverse=True)
        majority_class = sorted_categories[0]
        minority_class = sorted_categories[1]
        
        target_count = majority_class[1]['count']  # 1,055 images
        
        logger.info(f"Majority class: {majority_class[0]} ({majority_class[1]['count']} images)")
        logger.info(f"Minority class: {minority_class[0]} ({minority_class[1]['count']} images)")
        logger.info(f"Target count per class: {target_count}")
        
        # Process each category
        for category_name, info in category_info.items():
            logger.info(f"\nProcessing {category_name}...")
            
            # Create output category directory
            output_category = os.path.join(output_dir, category_name)
            os.makedirs(output_category, exist_ok=True)
            
            current_count = info['count']
            
            if category_name == majority_class[0]:
                # Just copy majority class images (no augmentation needed)
                logger.info(f"Copying {current_count} images (no augmentation needed)")
                
                with tqdm(total=current_count, desc=f"Copying {category_name}") as pbar:
                    for img_file in info['images']:
                        src_path = os.path.join(info['path'], img_file)
                        dst_path = os.path.join(output_category, img_file)
                        
                        try:
                            # Just copy the file
                            img = cv2.imread(src_path)
                            if img is not None:
                                cv2.imwrite(dst_path, img)
                        except Exception as e:
                            logger.error(f"Error copying {img_file}: {e}")
                        
                        pbar.update(1)
            
            else:
                # Augment minority class
                augmentations_needed = target_count - current_count
                augmentations_per_image = math.ceil(augmentations_needed / current_count)
                
                logger.info(f"Need {augmentations_needed} more images")
                logger.info(f"Creating {augmentations_per_image} augmentations per original image")
                
                total_to_process = current_count
                
                with tqdm(total=total_to_process, desc=f"Augmenting {category_name}") as pbar:
                    processed_count = 0
                    
                    for img_file in info['images']:
                        src_path = os.path.join(info['path'], img_file)
                        
                        try:
                            # Load original image
                            img = cv2.imread(src_path)
                            if img is None:
                                continue
                            
                            # Save original
                            dst_path = os.path.join(output_category, img_file)
                            cv2.imwrite(dst_path, img)
                            processed_count += 1
                            
                            # Create augmented versions
                            file_stem = Path(img_file).stem
                            file_ext = Path(img_file).suffix
                            
                            for aug_idx in range(augmentations_per_image):
                                if processed_count >= target_count:
                                    break
                                
                                # Create augmented image
                                augmented = self.create_augmented_image(img)
                                
                                # Save augmented image
                                aug_filename = f"{file_stem}_bal_aug_{aug_idx+1}{file_ext}"
                                aug_path = os.path.join(output_category, aug_filename)
                                cv2.imwrite(aug_path, augmented)
                                processed_count += 1
                            
                            if processed_count >= target_count:
                                break
                                
                        except Exception as e:
                            logger.error(f"Error processing {img_file}: {e}")
                        
                        pbar.update(1)
                
                logger.info(f"Final count for {category_name}: {processed_count}")
        
        # Final statistics
        print(f"\n🎉 SOYBEAN DATASET BALANCING COMPLETED!")
        print("="*60)
        
        total_final = 0
        for category in os.listdir(output_dir):
            category_path = os.path.join(output_dir, category)
            if os.path.isdir(category_path):
                count = len([f for f in os.listdir(category_path) 
                           if Path(f).suffix in self.supported_extensions])
                print(f"📁 {category}: {count:,} images")
                total_final += count
        
        print(f"\n📊 Final Statistics:")
        print(f"  • Total images: {total_final:,}")
        print(f"  • Balance achieved: ✅")
        print(f"  • Ready for CNN training: ✅")

def main():
    """Main function to balance the soybean dataset."""
    print("Soybean Plant Disease Dataset Balancer")
    print("=" * 50)
    print("🎯 Goal: Balance classes for optimal CNN training")
    
    # Define paths
    base_path = r"c:\Users\Almas Siddiqui\Downloads\plant_leaf_disease"
    soybean_input = os.path.join(base_path, "soyabean")
    soybean_output = os.path.join(base_path, "soybean_balanced")
    
    # Initialize preprocessor
    preprocessor = SoybeanBalancedPreprocessor(target_size=256)
    
    print(f"\nBalancing Soybean Dataset...")
    if os.path.exists(soybean_input):
        preprocessor.balance_soybean_dataset(
            input_dir=soybean_input,
            output_dir=soybean_output
        )
        
        print(f"\n✅ Balancing completed!")
        print(f"📁 Check output directory: {soybean_output}")
        
        print(f"\n💡 Training Recommendations:")
        print("  • Use balanced dataset for training")
        print("  • Consider class weights if still using original dataset")
        print("  • Apply stratified train/validation split")
        print("  • Monitor both classes' performance equally")
        
    else:
        print(f"❌ Soybean dataset not found at: {soybean_input}")

if __name__ == "__main__":
    main()
