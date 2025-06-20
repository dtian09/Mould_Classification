# Mould Classification and Segmentation

This repository contains scripts and tools for mould detection, classification, and segmentation using deep learning (ViT and U-Net) in PyTorch.

## Project Structure

- `generate_masks_from_yolo.py`  
  Generate binary segmentation masks from YOLO-format bounding box labels.

- `train_unet_segmentation.py`  
  Train a U-Net model for mould segmentation.

- `visualize_unet_segmentation.py`  
  Visualize U-Net segmentation predictions vs. ground truth.

- `label_mould_area.py`  
  Assign area-based class labels to images based on total mould coverage.

- `oversample_train_set.py`  
  Oversample the training set to balance class distribution.

- `count_classes.py`  
  Count the number of samples per class in a label file.

- `train_test.py`  
  Train and evaluate a ViT classifier with early stopping and class balancing.

## Installation

1. Clone this repository.
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Data

- Place your YOLO-format images and labels in the appropriate folders as referenced in the scripts.
- Segmentation masks will be generated in the `segmentation/` or `segmentation_masks/` directories.

## Usage

- **Generate masks:**  
  ```
  python generate_masks_from_yolo.py
  ```

- **Label images by mould area:**  
  ```
  python label_mould_area.py
  ```

- **Oversample training set:**  
  ```
  python oversample_train_set.py
  ```

- **Count class distribution:**  
  ```
  python count_classes.py
  ```

- **Train/test ViT classifier:**  
  ```
  python train_test.py
  ```
  
- **Train U-Net segmentation:**  
  ```
  python train_unet_segmentation.py
  ```

- **Visualize segmentation:**  
  ```
  python visualize_unet_segmentation.py
  ```

## Notes

- Masks: **Black = normal, White = mould**.
- Adjust paths and parameters in each script as needed for your dataset.

---

**Dependencies:** See