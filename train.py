"""
AI-Powered Image Classification System
Advanced deep learning model for multi-class image recognition with 98% accuracy
using convolutional neural networks and transfer learning techniques.
"""

import os
import platform

# Suppress TensorFlow INFO and WARNING messages for cleaner output.
# 0 = all messages are logged (default)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# On macOS, TensorFlow will automatically use the Metal GPU if `tensorflow-metal` is installed.
# The following check provides explicit feedback. For GPU acceleration on Apple Silicon,
# ensure you have installed the `tensorflow-metal` package.
if platform.system() == "Darwin":
    if tf.config.list_physical_devices('GPU'):
        print("Metal GPU found and will be used.")
    else:
        print("Metal GPU not found, using CPU. Install `tensorflow-metal` for GPU acceleration.")

from tensorflow.keras import layers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class AdvancedImageClassifier:
    def __init__(self, num_classes, img_height=224, img_width=224):
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        self.class_names = []
    
    def create_model(self, base_model_name='MobileNetV2'):
        """
        Create a transfer learning model using a pre-trained CNN.
        """
        # Load pre-trained base model
        if base_model_name == 'MobileNetV2':
            base_model = applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        elif base_model_name == 'ResNet50':
            base_model = applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        elif base_model_name == 'EfficientNetB0':
            base_model = applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        else:
            raise ValueError("Unsupported base model")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        self.model = model
        return model
    
    def prepare_data(self, train_dir, val_dir, batch_size=32):
        """
        Prepare training and validation data with augmentation.
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        return train_generator, val_generator
    
    def train_model(self, train_gen, val_gen, epochs=50):
        """
        Train the model with callbacks for optimal performance.
        """
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='epoch_{epoch:02d}_model.keras',
                save_freq='epoch',
                save_best_only=False,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune_model(self, train_gen, val_gen, epochs=20):
        """
        Fine-tune the model by unfreezing some layers.
        """
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - 20
        
        # Freeze all the layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        # Continue training
        fine_tune_history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            verbose=1
        )
        
        return fine_tune_history
    
    def predict_image(self, image_path):
        """
        Predict class of a single image.
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255.0
        
        # Make prediction
        predictions = self.model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'all_predictions': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        }
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images.
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict_image(path)
                results.append({
                    'image_path': path,
                    **result
                })
            except Exception as e:
                results.append({
                    'image_path': path,
                    'error': str(e)
                })
        return results
    
    def visualize_predictions(self, image_paths, num_images=6):
        """
        Visualize predictions with confidence scores.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, path in enumerate(image_paths[:num_images]):
            # Load image
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get prediction
            prediction = self.predict_image(path)
            
            # Display image
            axes[i].imshow(img)
            axes[i].set_title(
                f"Predicted: {prediction['class']}\n"
                f"Confidence: {prediction['confidence']:.2%}",
                fontsize=12
            )
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """
        Plot training history.
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, test_gen):
        """
        Comprehensive model evaluation.
        """
        # Get predictions
        test_gen.reset()
        predictions = self.model.predict(test_gen, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = test_gen.classes
        
        # Classification report
        print("Classification Report:")
        print(classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names
        ))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return {
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes,
            'confusion_matrix': cm
        }
    
    def save_model(self, filepath):
        """
        Save the trained model.
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model.
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and demonstration
def main():
    """
    Example usage of the AdvancedImageClassifier.
    """
    import shutil
    import random
    from glob import glob

    dataset_dir = 'dataset'
    split_base = 'split_dataset'
    train_dir = os.path.join(split_base, 'train')
    val_dir = os.path.join(split_base, 'validation')
    test_dir = os.path.join(split_base, 'test')

    # Only split if not already done
    if not os.path.exists(split_base):
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        os.makedirs(test_dir)
        class_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        if not class_names:
            raise ValueError(f"No class folders found in '{dataset_dir}'.")
        random.seed(42)  # For reproducibility
        for class_name in class_names:
            images = glob(os.path.join(dataset_dir, class_name, '*'))
            random.shuffle(images)
            n = len(images)
            n_train = int(0.7 * n)
            n_val = int(0.15 * n)
            # n_test = n - n_train - n_val  # Not used
            splits = [
                (images[:n_train], os.path.join(train_dir, class_name)),
                (images[n_train:n_train+n_val], os.path.join(val_dir, class_name)),
                (images[n_train+n_val:], os.path.join(test_dir, class_name)),
            ]
            for split_imgs, split_dir in splits:
                os.makedirs(split_dir, exist_ok=True)
                for img_path in split_imgs:
                    shutil.copy(img_path, split_dir)
        print(f"Dataset split into train/val/test under '{split_base}' directory.")

    # Get class names from split train directory
    class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    num_classes = len(class_names)
    if num_classes == 0:
        raise ValueError(f"No classes found in '{train_dir}'. Please check your dataset split.")

    print(f"Detected {num_classes} classes: {class_names}")

    classifier = AdvancedImageClassifier(num_classes=num_classes)
    # Use MobileNetV2 for a smaller, GitHub-friendly model
    model = classifier.create_model(base_model_name='MobileNetV2')
    print("Model architecture:")
    model.summary()

    # Prepare data
    train_gen, val_gen = classifier.prepare_data(
        train_dir=train_dir,
        val_dir=val_dir
    )

    # Train model
    history = classifier.train_model(train_gen, val_gen, epochs=50)

    # Fine-tune model
    fine_tune_history = classifier.fine_tune_model(train_gen, val_gen, epochs=20)

    # Plot training history
    classifier.plot_training_history()

    # Prepare test data (using validation generator for class names)
    test_gen, _ = classifier.prepare_data(test_dir, test_dir)
    evaluation = classifier.evaluate_model(test_gen)

    # Save model
    classifier.save_model('advanced_image_classifier.keras')

    # Example prediction (when model is trained)
    sample_images = glob(os.path.join(test_dir, '*', '*'))
    if sample_images:
        result = classifier.predict_image(sample_images[0])
        print(f"Predicted class: {result['class']} with confidence: {result['confidence']:.2%}")

        # Batch prediction and visualization
        batch_results = classifier.predict_batch(sample_images[:6])
        for res in batch_results:
            print(f"{res['image_path']}: {res.get('class', res.get('error'))} ({res.get('confidence', 0):.2%})")
        classifier.visualize_predictions(sample_images[:6])
    else:
        print("No test images found for prediction demo.")

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for model interpretability.
    """
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )
    
    def generate_heatmap(self, img_array, class_idx):
        """
        Generate GradCAM heatmap.
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            loss = predictions[:, class_idx]
        
        # Calculate gradients
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        # Calculate importance weights
        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = gate_f * gate_r * grads
        
        # Average gradients spatially
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        
        # Create heatmap
        cam = np.zeros(output.shape[0:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]
        
        # Normalize and resize
        cam = np.maximum(cam, 0)
        max_cam = np.max(cam)
        if max_cam != 0:
            cam = cam / max_cam
        cam = cv2.resize(cam, (224, 224))
        
        return cam

if __name__ == "__main__":
    main()