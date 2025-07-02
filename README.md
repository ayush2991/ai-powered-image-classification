# AI-Powered Image Classification System

This repository contains an advanced deep learning pipeline for multi-class image recognition using transfer learning. It supports training on datasets like Caltech101 and provides an interactive Streamlit app for model demonstration.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training (`train.py`)](#training-trainpy)
  - [Interactive Demo (`streamlit_app.py`)](#interactive-demo-streamlit_apppy)
- [Notes](#notes)

---

## Features

- Transfer learning with MobileNetV2 (default), ResNet50, or EfficientNetB0.
- Data augmentation and robust training callbacks.
- Model checkpointing after every epoch.
- Fine-tuning support for improved accuracy.
- Interactive Streamlit app for visual inference and class probability visualization.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training (`train.py`)

This script handles all steps from dataset preparation to model training and evaluation.

**Key Features:**
- Automatically splits your dataset into train/validation/test if not already split.
- Uses transfer learning with a pre-trained CNN (default: MobileNetV2 for smaller model size).
- Trains the classification head first, then fine-tunes the base model.
- Saves the best model and all epoch checkpoints.
- Evaluates the model and prints a classification report and confusion matrix.

**How to run:**

```bash
python train.py
```

**What happens:**
1. The script checks for a `dataset/` directory with subfolders for each class.
2. If not already split, it creates `split_dataset/train`, `split_dataset/validation`, and `split_dataset/test`.
3. Trains the model for 15 epochs (default) with only the head trainable.
4. Fine-tunes the model for 5 more epochs with some base layers unfrozen.
5. Saves the best model as `best_model.keras` and the final model as `advanced_image_classifier.keras`.
6. Prints evaluation metrics and shows training curves.

---

### Interactive Demo (`streamlit_app.py`)

A beautiful, interactive Streamlit app to showcase the trained model.

**Key Features:**
- Lets you browse sample images by class.
- Shows the model's prediction and confidence for each image.
- Displays a bar chart of the top class probabilities.
- Quick gallery for browsing multiple images.

**How to run:**

```bash
streamlit run streamlit_app.py
```

**What happens:**
- Loads the trained model (`best_model.keras` by default).
- Lets you select a class and image from the sidebar.
- Shows the selected image, predicted class, confidence, and a probability bar chart.
- Includes a gallery of sample images for quick browsing.

---

## Notes

- The default model is MobileNetV2 for fast training and small file size (suitable for GitHub).
- You can switch to ResNet50 or EfficientNetB0 by changing the `base_model_name` argument in `train.py`.
- For best results, use a well-organized dataset with one subfolder per class under `dataset/`.

---

**Enjoy exploring and deploying your own image classifier!**
