# Face Mask Detection Model

This repository contains code and resources for training a Face Mask Detection model using Hugging Face's `transformers` library. The goal is to build a custom image classification model to detect whether a person is wearing a mask or not.

## Problem Overview

The goal is to classify images into two categories:
1. **With Mask**
2. **Without Mask**

We use the Vision Transformer (ViT) model pre-trained on a large dataset to fine-tune it for our specific task.

## Discussions and Prompts Summary

### 1. Initial Setup and Model Selection

We started by choosing a pre-trained Vision Transformer (ViT) model from Hugging Face (`google/vit-base-patch16-224-in21k`). The model was then fine-tuned on a custom dataset containing images of people wearing masks and not wearing masks. 

**Prompt**:  
- How to use Hugging Face's pre-trained Vision Transformer model for image classification.
- How to prepare and process images for training using `transformers` and PyTorch.

### 2. Code Issues and Troubleshooting

During the training process, the following issues were identified:
- **CUDA Library Missing**: A warning was displayed regarding missing `cudart64_110.dll`, which can be ignored if no GPU is available.
- **Data Loading**: We explored ways to handle slow data loading and potential memory bottlenecks.
- **Model Weights Not Initialized**: Some weights of the `ViTForImageClassification` model were not initialized properly, which is common when fine-tuning models for custom tasks.
- **Training Stuck**: Training sometimes got stuck, which was addressed by adjusting the data loading, batch size, and ensuring proper CPU or GPU setup.

**Prompt**:  
- Debugging and handling model loading issues.
- Optimizing the training pipeline for different environments.

### 3. Training Process

The model was trained using Hugging Face's `Trainer` API. The `TrainingArguments` were adjusted to fit the problem's requirements. The training pipeline involved:
- Loading and processing a custom dataset of images.
- Fine-tuning the Vision Transformer model on the dataset.
- Evaluating the model performance.

**Prompt**:  
- How to set up and train the Vision Transformer model with custom data.
- Adjusting `TrainingArguments` for optimal training performance.

### 4. Troubleshooting Training Issues

The training process showed a potential bottleneck during data loading or model processing. We tried to:
- Check if the system’s memory usage was too high.
- Ensure that the correct device (CPU or GPU) was used for training.
- Use smaller batch sizes to avoid memory issues.

**Prompt**:  
- How to monitor and troubleshoot training progress.
- Handling slow or stuck training processes.

## Dataset

The dataset used in this project consists of images of individuals with and without face masks. These images are located in the following directories:
- `with_mask_dir`: Contains images of people wearing masks.
- `without_mask_dir`: Contains images of people not wearing masks.

### 4. Requirements

To run this project, you need the following dependencies:

- Python 3.10+
- transformers library
- torch library
- huggingface_hub
- PIL (for image processing)

### Known Issues and Future Improvements
- Slow Data Loading: Experiment with smaller datasets or optimize image pre-processing pipelines.
- Memory Issues: Ensure sufficient memory for training, or use smaller batch sizes.
- Fine-Tuning: The model may require additional fine-tuning for better performance depending on the dataset size and quality.
