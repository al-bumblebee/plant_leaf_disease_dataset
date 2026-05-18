# Plant Leaf Disease Image Preprocessor

A comprehensive Python script for preprocessing plant leaf disease images (Rice and Sugarcane) for CNN training.

## Features

✅ **Resize with Padding**: Converts images of different resolutions to 256x256 while maintaining aspect ratio  
✅ **Data Augmentation**: Random rotation, flip, brightness/contrast adjustment, and zoom  
✅ **Normalization**: Pixel values normalized to [0,1] range  
✅ **Multi-format Support**: Works with JPEG and PNG files  
✅ **Progress Tracking**: Real-time progress bars and detailed logging  
✅ **Error Handling**: Robust error handling with detailed reporting  

## Dataset Structure

### Input Structure
```
plant_leaf_disease/
├── rice_leaf_diseases/
│   ├── Bacterial leaf blight/
│   ├── Brown spot/
│   └── Leaf smut/
└── Sugarcane Leaf Disease Dataset/
    ├── Healthy/
    ├── Mosaic/
    ├── RedRot/
    ├── Rust/
    └── Yellow/
```

### Output Structure
```
plant_leaf_disease/
├── rice_processed/
│   ├── Bacterial leaf blight/    # 120 images (40 original + 80 augmented)
│   ├── Brown spot/               # 120 images (40 original + 80 augmented)
│   └── Leaf smut/                # 120 images (40 original + 80 augmented)
└── sugarcane_processed/
    ├── Healthy/                  # 1,566 images (522 original + 1,044 augmented)
    ├── Mosaic/                   # 1,386 images (462 original + 924 augmented)
    ├── RedRot/                   # 1,554 images (518 original + 1,036 augmented)
    ├── Rust/                     # 1,542 images (514 original + 1,028 augmented)
    └── Yellow/                   # 1,515 images (505 original + 1,010 augmented)
```

## Installation

1. **Install Python 3.7+**

2. **Install required packages:**
```bash
pip install opencv-python numpy tqdm
```

3. **Or use the requirements file:**
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from plant_leaf_preprocessor import PlantLeafPreprocessor

# Initialize preprocessor
preprocessor = PlantLeafPreprocessor(target_size=256, padding_color=(128, 128, 128))

# Process dataset
preprocessor.preprocess_dataset(
    input_dir="path/to/input/dataset",
    output_dir="path/to/output/dataset",
    apply_augmentation=True,
    augmentation_factor=2,
    augment_prob=0.5
)
```

### Command Line Usage
```bash
python plant_leaf_preprocessor.py --input_dir "input/path" --output_dir "output/path" --augment --aug_factor 2
```

### Run with Default Settings
Simply run the script to process both datasets with default settings:
```bash
python plant_leaf_preprocessor.py
```

## Processing Results

### Rice Leaf Disease Dataset
- **Original Images**: 120
- **After Augmentation**: 360 images (3x increase)
- **Categories**: 3 (Bacterial leaf blight, Brown spot, Leaf smut)
- **Processing Time**: ~10 seconds

### Sugarcane Leaf Disease Dataset
- **Original Images**: 2,521
- **After Augmentation**: 7,563 images (3x increase)
- **Categories**: 5 (Healthy, Mosaic, RedRot, Rust, Yellow)
- **Processing Time**: ~2.5 minutes

## Image Processing Details

### 1. Resize with Padding
- Maintains original aspect ratio
- Scales longer side to 256 pixels
- Pads shorter side with gray pixels (128, 128, 128)
- Results in square 256x256 images

### 2. Data Augmentation (Optional)
- **Rotation**: Random rotation (-30° to +30°)
- **Flip**: Random horizontal/vertical flipping
- **Brightness/Contrast**: Random adjustment (0.8x to 1.2x)
- **Zoom**: Random zoom (0.9x to 1.1x)
- **Probability**: Each augmentation applied with 50% probability

### 3. Normalization
- Pixel values normalized to [0, 1] range
- Converted back to [0, 255] for saving

## File Structure

```
plant_leaf_disease/
├── plant_leaf_preprocessor.py      # Main preprocessing script
├── analyze_results_simple.py       # Results analysis script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── rice_processed/                 # Processed rice images
├── sugarcane_processed/            # Processed sugarcane images
├── rice_leaf_diseases/             # Original rice dataset
└── Sugarcane Leaf Disease Dataset/ # Original sugarcane dataset
```

## Verification

Run the analysis script to verify processing results:
```bash
python analyze_results_simple.py
```

This will show:
- Image count per category
- Image size consistency verification
- Augmentation statistics
- Pixel value ranges

## Next Steps for CNN Training

1. **Import processed datasets** into your ML framework (TensorFlow/PyTorch)
2. **Split data** into train/validation/test sets (e.g., 70/15/15)
3. **Create data loaders** with batch processing
4. **Build CNN model** architecture
5. **Train the model** using the preprocessed images
6. **Use folder names** as class labels

### Example TensorFlow Data Loading
```python
import tensorflow as tf

# Create dataset from directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    'rice_processed',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'rice_processed',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=32
)
```

## Performance

- **Rice Dataset**: ~12 images/second
- **Sugarcane Dataset**: ~16 images/second
- **Memory Usage**: Low (processes images one at a time)
- **Storage**: ~3x increase due to augmentation

## Error Handling

The script includes comprehensive error handling:
- Skips corrupted or unreadable images
- Logs all errors with file paths
- Continues processing even if individual images fail
- Provides detailed success/error statistics

## Customization

### Adjust Target Size
```python
preprocessor = PlantLeafPreprocessor(target_size=224)  # For ResNet compatibility
```

### Change Padding Color
```python
preprocessor = PlantLeafPreprocessor(padding_color=(0, 0, 0))  # Black padding
```

### Modify Augmentation Parameters
```python
preprocessor.preprocess_dataset(
    input_dir="input/path",
    output_dir="output/path",
    apply_augmentation=True,
    augmentation_factor=3,    # Create 3 augmented versions
    augment_prob=0.7         # 70% chance for each augmentation
)
```

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the preprocessor.
