# Building Mould Area Coverage Classification and Segmentation

This repository contains scripts and tools for building mould area coverage classification, and segmentation using deep learning (**ViT** and **U-Net**) in PyTorch.
- dataset: [YOLOv7 data set of Mould Detection Single Label Computer Vision Project](https://universe.roboflow.com/research-placement/mould-detection-single-label)
- For each image, the **total normalized mould area** is calculated as the sum of the areas of all bounding boxes in YOLO label files:

  \[
  \text{Total Mould Area} = \sum_{i=1}^{N} (\text{width}_i \times \text{height}_i)
  \]
  
  where `width` and `height` are the normalized values (between 0 and 1) from each bounding box line in the YOLO label file; N is the number of mould areas in the image.

**Area categories:**
- `0`: normal (no mould)
- `1`: small or medium (0 < area ≤ 0.15)
- `2`: large (0.15 < area ≤ 0.3)
- `3`: extra large (area > 0.3)  
      
## Code

- `label_mould_area.py`  
  Assign area-based class labels to images based on total mould coverage.

- `oversample_train_set.py`  
  Oversample the training set to balance class distribution.

- `count_classes.py`  
  Count the number of samples per class in a label file.

- `train_test.py`  
  Train and evaluate a ViT classifier with early stopping and class balancing.

- `generate_masks_from_yolo.py`  
  Generate binary segmentation masks from YOLO-format bounding box labels.

- `train_unet_segmentation.py`  
  Train a U-Net model for mould segmentation.

- `visualize_unet_segmentation.py`  
  Visualize U-Net segmentation predictions vs. ground truth.
  
## Installation

1. Clone this repository.
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Data

- Place your [YOLOv7 dataset](https://universe.roboflow.com/research-placement/mould-detection-single-label) folder in the folder `Mould_Classification_Segmentation/`.
- Segmentation masks will be generated in the `segmentation_masks/` directories.

## Usage: Building Mould Area Coverage Classification Using Vision Transformer (ViT)

- Step 1: **Label images by mould area:**  
  ```
  python label_mould_area.py
  ```
  
- Step 2: **Count class distribution:**  
  ```
  python count_classes.py
  ```
  
- Step 3: **Oversample training set:**  
  ```
  python oversample_train_set.py
  ```

- Step 4: **Train/test ViT classifier:**  
  ```
  python train_test.py
  ```
## Usage: Building Mould Segmentation Using U-Net

- Step 1: **Generate masks:**  
  ```
  python generate_masks_from_yolo.py
  ```
    
- Step 2: **Train U-Net segmentation:**  
  ```
  python train_unet_segmentation.py
  ```

- Step 3: **Visualize segmentation:**  
  ```
  python visualize_unet_segmentation.py
  ```

## Notes

- Masks: **Black = normal, White = mould**.
- Adjust paths and parameters in each script as needed for your dataset.
