# Iranian Car Fine-Grained Image Classification (FGIC)

## Overview

This project presents a Fine-Grained Image Classification (FGIC) system tailored for Iranian car models. Given the visual similarity of various Iranian vehicles, this task focuses on identifying subtle features that distinguish between classes using deep learning.

The project applies a custom Convolutional Neural Network (CNN) model, enhanced by data augmentation and regularization, to achieve strong classification performance.

## Contents

- `Iranian_car_FGIC.ipynb`: Full project notebook including:
  - Image loading and visualization
  - Data augmentation and preprocessing
  - Custom CNN design
  - Training and validation
  - Performance evaluation and visualizations
- Output figures and graphs:
  - Accuracy & loss plots
  - Confusion matrix
  - Model predictions

## Dataset

The dataset contains labeled images of Iranian cars, categorized by make and model. Data augmentation techniques such as flipping, zooming, and rotation were applied to increase dataset diversity.

## Data Preprocessing & Augmentation

Before training the model, all images were preprocessed to ensure consistency and improve model performance:

- **Resizing**: All images were resized to a uniform size to match model input dimensions.
- **Normalization**: Pixel values were scaled to the [0, 1] range for stable gradient flow.
- **Splitting**: The dataset was split into training and validation sets using a stratified approach to preserve class distribution.
- **Cleaning**: Duplicate or corrupted images were removed from the dataset.

To improve the model's generalization and combat overfitting, several **data augmentation techniques** were applied:

- Horizontal and vertical flipping
- Random rotation
- Zoom and brightness adjustments
- Cropping and shifting

These augmentations simulate real-world variability and help the model learn robust features across different views and lighting conditions.


## Model Architecture

The CNN model includes:
- Multiple convolution layers with ReLU activation
- MaxPooling layers
- Dropout for regularization
- Fully connected layers leading to a softmax classifier

## Evaluation

The model was evaluated on accuracy and loss across epochs, confusion matrix, and visual inspection of predictions. Results indicate the model effectively distinguishes between visually similar car types.

## Final Results

- Validation Accuracy: *~89%*
- Number of Classes: *13 (Iranian car models)*
- Training Configuration:
  - Hardware: Trained on CPU
  - Epochs: 5
  - Initial Learning Rate: 0.0010
  - Learning Rate Adjustment: Reduced to 1.0000e-04 after epoch 2 using a custom scheduler
  - Batch Size: 32

## Testing result

- accuracy                         0.8919      1323
- macro avg     0.8922    0.8941    0.8922      1323
- weighted avg     0.8935    0.8919    0.8917      1323

- Precision(macro): 0.8922
- Recall(macro):    0.8941
- F1 Score(macro):  0.8922
- F1 Score(micro):  0.8919

## Visualizations

### Accuracy and Loss

![Training Accuracy and Loss](download.png)

### Confusion Matrix

![Confusion Matrix](download(1).png)

### Sample Predictions

![Sample Predictions](download(2).png)

## How to Run

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
