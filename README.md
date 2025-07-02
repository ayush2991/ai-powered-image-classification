# AI-Powered Image Classification System

This repository contains an advanced deep learning pipeline for multi-class image recognition using transfer learning. It supports training on datasets like Caltech101 and provides an interactive Streamlit app for model demonstration.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training (`train.py`)](#training-trainpy)
  - [Interactive Demo (`streamlit_app.py`)](#interactive-demo-streamlit_apppy)
- [Training Results](#training-results)
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

## Training Results

### Training and Validation Loss

![Training and Validation Loss](train_val_loss.png)

### Training Logs

<details>
<summary>Click to expand logs.txt</summary>

```
(.venv) aayushagarwal-mac:ai-powered-image-classification aayushagarwal$ python3 train.py 
Metal GPU found and will be used.
Detected 102 classes: ['gerenuk', 'hawksbill', 'headphone', 'ant', 'butterfly', 'lamp', 'strawberry', 'water_lilly', 'chandelier', 'dragonfly', 'crab', 'pagoda', 'dollar_bill', 'emu', 'inline_skate', 'platypus', 'dalmatian', 'cup', 'airplanes', 'joshua_tree', 'cougar_body', 'grand_piano', 'trilobite', 'brontosaurus', 'wild_cat', 'pigeon', 'dolphin', 'soccer_ball', 'wrench', 'scorpion', 'flamingo_head', 'nautilus', 'accordion', 'cougar_face', 'pyramid', 'camera', 'barrel', 'schooner', 'cellphone', 'panda', 'revolver', 'lobster', 'menorah', 'lotus', 'stapler', 'crocodile', 'chair', 'helicopter', 'minaret', 'starfish', 'ceiling_fan', 'ketch', 'mayfly', 'wheelchair', 'bass', 'yin_yang', 'crocodile_head', 'saxophone', 'beaver', 'mandolin', 'bonsai', 'Leopards', 'car_side', 'ibis', 'electric_guitar', 'kangaroo', 'stegosaurus', 'ferry', 'snoopy', 'umbrella', 'rhino', 'okapi', 'watch', 'brain', 'gramophone', 'scissors', 'rooster', 'cannon', 'binocular', 'anchor', 'octopus', 'buddha', 'laptop', 'windsor_chair', 'hedgehog', 'pizza', 'euphonium', 'stop_sign', 'Motorbikes', 'sea_horse', 'flamingo', 'BACKGROUND_Google', 'ewer', 'garfield', 'crayfish', 'Faces_easy', 'Faces', 'sunflower', 'llama', 'elephant', 'tick', 'metronome']
Model architecture:
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ mobilenetv2_1.00_224 (Functional)    │ (None, 7, 7, 1280)          │       2,257,984 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 1280)                │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 1280)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 512)                 │         655,872 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 512)                 │           2,048 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 512)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 256)                 │         131,328 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 256)                 │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 102)                 │          26,214 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 3,074,470 (11.73 MB)
 Trainable params: 814,950 (3.11 MB)
 Non-trainable params: 2,259,520 (8.62 MB)
Found 6352 images belonging to 102 classes.
Found 1326 images belonging to 102 classes.
/Users/aayushagarwal/projects/ai-powered-image-classification/.venv/lib/python3.12/site-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 132ms/step - accuracy: 0.3423 - loss: 3.2144 - top_5_accuracy: 0.4995    
Epoch 1: val_accuracy improved from -inf to 0.82278, saving model to best_model.keras

Epoch 1: saving model to epoch_01_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 34s 156ms/step - accuracy: 0.3431 - loss: 3.2100 - top_5_accuracy: 0.5004 - val_accuracy: 0.8228 - val_loss: 0.7543 - val_top_5_accuracy: 0.9555 - learning_rate: 0.0010
Epoch 2/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 131ms/step - accuracy: 0.7092 - loss: 1.2087 - top_5_accuracy: 0.8941 
Epoch 2: val_accuracy improved from 0.82278 to 0.87029, saving model to best_model.keras

Epoch 2: saving model to epoch_02_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 145ms/step - accuracy: 0.7093 - loss: 1.2082 - top_5_accuracy: 0.8942 - val_accuracy: 0.8703 - val_loss: 0.5407 - val_top_5_accuracy: 0.9661 - learning_rate: 0.0010
Epoch 3/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 132ms/step - accuracy: 0.7754 - loss: 0.8544 - top_5_accuracy: 0.9374 
Epoch 3: val_accuracy improved from 0.87029 to 0.88763, saving model to best_model.keras

Epoch 3: saving model to epoch_03_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 145ms/step - accuracy: 0.7754 - loss: 0.8544 - top_5_accuracy: 0.9374 - val_accuracy: 0.8876 - val_loss: 0.4327 - val_top_5_accuracy: 0.9796 - learning_rate: 0.0010
Epoch 4/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 132ms/step - accuracy: 0.7953 - loss: 0.7627 - top_5_accuracy: 0.9482 
Epoch 4: val_accuracy did not improve from 0.88763

Epoch 4: saving model to epoch_04_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 145ms/step - accuracy: 0.7953 - loss: 0.7627 - top_5_accuracy: 0.9482 - val_accuracy: 0.8816 - val_loss: 0.4448 - val_top_5_accuracy: 0.9698 - learning_rate: 0.0010
Epoch 5/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 133ms/step - accuracy: 0.8200 - loss: 0.6581 - top_5_accuracy: 0.9590 
Epoch 5: val_accuracy improved from 0.88763 to 0.89819, saving model to best_model.keras

Epoch 5: saving model to epoch_05_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 146ms/step - accuracy: 0.8200 - loss: 0.6581 - top_5_accuracy: 0.9590 - val_accuracy: 0.8982 - val_loss: 0.3881 - val_top_5_accuracy: 0.9789 - learning_rate: 0.0010
Epoch 6/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 131ms/step - accuracy: 0.8296 - loss: 0.6124 - top_5_accuracy: 0.9644 
Epoch 6: val_accuracy improved from 0.89819 to 0.89894, saving model to best_model.keras

Epoch 6: saving model to epoch_06_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 145ms/step - accuracy: 0.8296 - loss: 0.6124 - top_5_accuracy: 0.9644 - val_accuracy: 0.8989 - val_loss: 0.3721 - val_top_5_accuracy: 0.9811 - learning_rate: 0.0010
Epoch 7/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 132ms/step - accuracy: 0.8348 - loss: 0.5642 - top_5_accuracy: 0.9667 
Epoch 7: val_accuracy did not improve from 0.89894

Epoch 7: saving model to epoch_07_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 145ms/step - accuracy: 0.8347 - loss: 0.5643 - top_5_accuracy: 0.9667 - val_accuracy: 0.8967 - val_loss: 0.3718 - val_top_5_accuracy: 0.9796 - learning_rate: 0.0010
Epoch 8/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 137ms/step - accuracy: 0.8386 - loss: 0.5747 - top_5_accuracy: 0.9663 
Epoch 8: val_accuracy did not improve from 0.89894

Epoch 8: saving model to epoch_08_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 30s 151ms/step - accuracy: 0.8387 - loss: 0.5745 - top_5_accuracy: 0.9663 - val_accuracy: 0.8982 - val_loss: 0.3496 - val_top_5_accuracy: 0.9857 - learning_rate: 0.0010
Epoch 9/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 143ms/step - accuracy: 0.8408 - loss: 0.5507 - top_5_accuracy: 0.9676 
Epoch 9: val_accuracy improved from 0.89894 to 0.90724, saving model to best_model.keras

Epoch 9: saving model to epoch_09_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 31s 157ms/step - accuracy: 0.8408 - loss: 0.5507 - top_5_accuracy: 0.9676 - val_accuracy: 0.9072 - val_loss: 0.3313 - val_top_5_accuracy: 0.9872 - learning_rate: 0.0010
Epoch 10/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 133ms/step - accuracy: 0.8646 - loss: 0.5163 - top_5_accuracy: 0.9723 
Epoch 10: val_accuracy did not improve from 0.90724

Epoch 10: saving model to epoch_10_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 146ms/step - accuracy: 0.8645 - loss: 0.5163 - top_5_accuracy: 0.9723 - val_accuracy: 0.8929 - val_loss: 0.3450 - val_top_5_accuracy: 0.9834 - learning_rate: 0.0010
Epoch 11/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 130ms/step - accuracy: 0.8652 - loss: 0.4972 - top_5_accuracy: 0.9708 
Epoch 11: val_accuracy did not improve from 0.90724

Epoch 11: saving model to epoch_11_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 28s 142ms/step - accuracy: 0.8651 - loss: 0.4972 - top_5_accuracy: 0.9708 - val_accuracy: 0.9057 - val_loss: 0.3470 - val_top_5_accuracy: 0.9834 - learning_rate: 0.0010
Epoch 12/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 131ms/step - accuracy: 0.8682 - loss: 0.4617 - top_5_accuracy: 0.9774 
Epoch 12: val_accuracy did not improve from 0.90724

Epoch 12: saving model to epoch_12_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 144ms/step - accuracy: 0.8682 - loss: 0.4617 - top_5_accuracy: 0.9774 - val_accuracy: 0.9012 - val_loss: 0.3335 - val_top_5_accuracy: 0.9819 - learning_rate: 0.0010
Epoch 13/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 144ms/step - accuracy: 0.8625 - loss: 0.4484 - top_5_accuracy: 0.9775 
Epoch 13: val_accuracy did not improve from 0.90724

Epoch 13: saving model to epoch_13_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 31s 158ms/step - accuracy: 0.8625 - loss: 0.4485 - top_5_accuracy: 0.9775 - val_accuracy: 0.8952 - val_loss: 0.3479 - val_top_5_accuracy: 0.9864 - learning_rate: 0.0010
Epoch 14/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 142ms/step - accuracy: 0.8638 - loss: 0.4432 - top_5_accuracy: 0.9748 
Epoch 14: val_accuracy did not improve from 0.90724

Epoch 14: saving model to epoch_14_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 31s 154ms/step - accuracy: 0.8638 - loss: 0.4433 - top_5_accuracy: 0.9748 - val_accuracy: 0.8974 - val_loss: 0.3529 - val_top_5_accuracy: 0.9842 - learning_rate: 0.0010
Epoch 15/15
199/199 ━━━━━━━━━━━━━━━━━━━━ 0s 132ms/step - accuracy: 0.8711 - loss: 0.4275 - top_5_accuracy: 0.9789 
Epoch 15: val_accuracy did not improve from 0.90724

Epoch 15: saving model to epoch_15_model.keras
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 144ms/step - accuracy: 0.8712 - loss: 0.4274 - top_5_accuracy: 0.9789 - val_accuracy: 0.9042 - val_loss: 0.3216 - val_top_5_accuracy: 0.9849 - learning_rate: 2.0000e-04
Epoch 1/5
199/199 ━━━━━━━━━━━━━━━━━━━━ 38s 159ms/step - accuracy: 0.8115 - loss: 0.6573 - top_5_accuracy: 0.9584 - val_accuracy: 0.8959 - val_loss: 0.3889 - val_top_5_accuracy: 0.9729
Epoch 2/5
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 147ms/step - accuracy: 0.8293 - loss: 0.5942 - top_5_accuracy: 0.9645 - val_accuracy: 0.9012 - val_loss: 0.3782 - val_top_5_accuracy: 0.9751
Epoch 3/5
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 147ms/step - accuracy: 0.8419 - loss: 0.5469 - top_5_accuracy: 0.9669 - val_accuracy: 0.9020 - val_loss: 0.3599 - val_top_5_accuracy: 0.9789
Epoch 4/5
199/199 ━━━━━━━━━━━━━━━━━━━━ 30s 148ms/step - accuracy: 0.8522 - loss: 0.5308 - top_5_accuracy: 0.9685 - val_accuracy: 0.9035 - val_loss: 0.3467 - val_top_5_accuracy: 0.9796
Epoch 5/5
199/199 ━━━━━━━━━━━━━━━━━━━━ 29s 147ms/step - accuracy: 0.8533 - loss: 0.5142 - top_5_accuracy: 0.9697 - val_accuracy: 0.9050 - val_loss: 0.3336 - val_top_5_accuracy: 0.9834
2025-07-02 01:13:46.169 Python[65224:4133355] The class 'NSSavePanel' overrides the method identifier.  This method is implemented by class 'NSWindow'
Found 1466 images belonging to 102 classes.
Found 1466 images belonging to 102 classes.
46/46 ━━━━━━━━━━━━━━━━━━━━ 9s 171ms/step
Classification Report:
                   precision    recall  f1-score   support

BACKGROUND_Google       0.08      0.06      0.06        71
            Faces       0.09      0.09      0.09        66
       Faces_easy       0.03      0.03      0.03        66
         Leopards       0.03      0.03      0.03        30
       Motorbikes       0.05      0.05      0.05       121
        accordion       0.00      0.00      0.00         9
        airplanes       0.02      0.03      0.02       120
           anchor       0.00      0.00      0.00         7
              ant       0.00      0.00      0.00         7
           barrel       0.00      0.00      0.00         8
             bass       0.00      0.00      0.00         9
           beaver       0.00      0.00      0.00         8
        binocular       0.00      0.00      0.00         6
           bonsai       0.00      0.00      0.00        20
            brain       0.00      0.00      0.00        16
     brontosaurus       0.00      0.00      0.00         7
           buddha       0.00      0.00      0.00        14
        butterfly       0.08      0.07      0.07        15
           camera       0.00      0.00      0.00         8
           cannon       0.00      0.00      0.00         7
         car_side       0.00      0.00      0.00        19
      ceiling_fan       0.00      0.00      0.00         8
        cellphone       0.00      0.00      0.00        10
            chair       0.00      0.00      0.00        10
       chandelier       0.00      0.00      0.00        17
      cougar_body       0.00      0.00      0.00         8
      cougar_face       0.00      0.00      0.00        11
             crab       0.00      0.00      0.00        12
         crayfish       0.00      0.00      0.00        11
        crocodile       0.00      0.00      0.00         8
   crocodile_head       0.00      0.00      0.00         9
              cup       0.00      0.00      0.00        10
        dalmatian       0.00      0.00      0.00        11
      dollar_bill       0.00      0.00      0.00         9
          dolphin       0.00      0.00      0.00        11
        dragonfly       0.00      0.00      0.00        11
  electric_guitar       0.00      0.00      0.00        12
         elephant       0.00      0.00      0.00        11
              emu       0.00      0.00      0.00         9
        euphonium       0.00      0.00      0.00        11
             ewer       0.00      0.00      0.00        14
            ferry       0.00      0.00      0.00        11
         flamingo       0.00      0.00      0.00        11
    flamingo_head       0.00      0.00      0.00         8
         garfield       0.00      0.00      0.00         6
          gerenuk       0.00      0.00      0.00         6
       gramophone       0.00      0.00      0.00         9
      grand_piano       0.06      0.06      0.06        16
        hawksbill       0.00      0.00      0.00        15
        headphone       0.00      0.00      0.00         7
         hedgehog       0.00      0.00      0.00         9
       helicopter       0.00      0.00      0.00        14
             ibis       0.00      0.00      0.00        12
     inline_skate       0.00      0.00      0.00         6
      joshua_tree       0.08      0.09      0.08        11
         kangaroo       0.00      0.00      0.00        14
            ketch       0.05      0.06      0.05        18
             lamp       0.00      0.00      0.00        10
           laptop       0.00      0.00      0.00        13
            llama       0.00      0.00      0.00        13
          lobster       0.00      0.00      0.00         7
            lotus       0.00      0.00      0.00        11
         mandolin       0.00      0.00      0.00         7
           mayfly       0.20      0.17      0.18         6
          menorah       0.00      0.00      0.00        14
        metronome       0.00      0.00      0.00         6
          minaret       0.00      0.00      0.00        12
         nautilus       0.00      0.00      0.00         9
          octopus       0.00      0.00      0.00         6
            okapi       0.00      0.00      0.00         7
           pagoda       0.00      0.00      0.00         8
            panda       0.00      0.00      0.00         7
           pigeon       0.00      0.00      0.00         8
            pizza       0.00      0.00      0.00         9
         platypus       0.00      0.00      0.00         6
          pyramid       0.00      0.00      0.00        10
         revolver       0.00      0.00      0.00        13
            rhino       0.00      0.00      0.00        10
          rooster       0.00      0.00      0.00         8
        saxophone       0.00      0.00      0.00         6
         schooner       0.00      0.00      0.00        10
         scissors       0.00      0.00      0.00         7
         scorpion       0.06      0.07      0.07        14
        sea_horse       0.00      0.00      0.00        10
           snoopy       0.00      0.00      0.00         6
      soccer_ball       0.00      0.00      0.00        11
          stapler       0.00      0.00      0.00         8
         starfish       0.00      0.00      0.00        14
      stegosaurus       0.00      0.00      0.00        10
        stop_sign       0.00      0.00      0.00        11
       strawberry       0.00      0.00      0.00         6
        sunflower       0.00      0.00      0.00        14
             tick       0.00      0.00      0.00         8
        trilobite       0.00      0.00      0.00        14
         umbrella       0.00      0.00      0.00        12
            watch       0.02      0.03      0.03        37
      water_lilly       0.00      0.00      0.00         7
       wheelchair       0.00      0.00      0.00        10
         wild_cat       0.00      0.00      0.00         6
    windsor_chair       0.00      0.00      0.00         9
           wrench       0.00      0.00      0.00         7
         yin_yang       0.00      0.00      0.00         9

         accuracy                           0.02      1466
        macro avg       0.01      0.01      0.01      1466
     weighted avg       0.02      0.02      0.02      1466

Model saved to advanced_image_classifier.keras
1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step
Predicted class: gerenuk with confidence: 81.19%
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 49ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step
split_dataset/test/gerenuk/image_0031.jpg: gerenuk (81.19%)
split_dataset/test/gerenuk/image_0018.jpg: gerenuk (44.44%)
split_dataset/test/gerenuk/image_0007.jpg: gerenuk (92.47%)
split_dataset/test/gerenuk/image_0010.jpg: gerenuk (98.90%)
split_dataset/test/gerenuk/image_0014.jpg: gerenuk (93.27%)
split_dataset/test/gerenuk/image_0003.jpg: gerenuk (99.00%)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
```
</details>

---

## Notes

- The default model is MobileNetV2 for fast training and small file size (suitable for GitHub).
- You can switch to ResNet50 or EfficientNetB0 by changing the `base_model_name` argument in `train.py`.
- For best results, use a well-organized dataset with one subfolder per class under `dataset/`.

---

**Enjoy exploring and deploying your own image classifier!**
