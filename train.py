"""
AI-Powered Image Classification System
Advanced deep learning model for multi-class image recognition using transfer learning.
"""

import os
import random
from glob import glob
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ------------------- CONFIGURATION -------------------
IMG_SIZE = (300, 200)
BATCH_SIZE = 64
NUM_WORKERS = 2  # Increase if you have more CPU cores
EPOCHS = 10
FINE_TUNE_EPOCHS = 5
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-4
PLOT_HISTORY = False  # Set True to enable training/validation plots

# ------------------- DEVICE SETUP -------------------
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class AdvancedImageClassifierTorch(nn.Module):
    """Transfer learning classifier supporting MobileNetV2 and ResNet50."""
    def __init__(self, num_classes, base_model_name='mobilenet_v2'):
        super().__init__()
        if base_model_name == 'mobilenet_v2':
            self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            in_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Identity()
        elif base_model_name == 'resnet50':
            self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.base.fc.in_features
            self.base.fc = nn.Identity()
        else:
            raise ValueError("Unsupported base model")
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x

def prepare_data(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Prepare train/validation dataloaders with augmentations."""
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'validation'), transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    class_names = train_dataset.classes
    return train_loader, val_loader, class_names

def train_model(model, train_loader, val_loader, num_epochs=EPOCHS, lr=LEARNING_RATE, save_path='best_model.pt', plot_history=PLOT_HISTORY):
    """Train model and optionally plot/save training history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, min_lr=1e-7)
    best_val_acc = 0
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        train_loss_hist.append(epoch_loss)
        train_acc_hist.append(epoch_acc)

        # Validation
        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)
        val_loss_epoch = val_loss / val_total
        val_acc_epoch = val_corrects / val_total
        val_loss_hist.append(val_loss_epoch)
        val_acc_hist.append(val_acc_epoch)
        scheduler.step(val_loss_epoch)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | Val Loss: {val_loss_epoch:.4f}, Acc: {val_acc_epoch:.4f}")

        # Save best model
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        # Save model every epoch
        torch.save(model.state_dict(), f"epoch_{epoch+1:02d}_model.pt")

    if plot_history:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_hist, label='Train Loss')
        plt.plot(val_loss_hist, label='Val Loss')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_acc_hist, label='Train Acc')
        plt.plot(val_acc_hist, label='Val Acc')
        plt.title('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig('train_val_loss.png')
        plt.show()

def evaluate_model(model, data_loader, class_names):
    """Print classification report and plot confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def main():
    """Main training and evaluation pipeline."""
    # Split dataset if not already split
    dataset_dir = 'dataset'
    split_base = 'split_dataset'
    train_dir = os.path.join(split_base, 'train')
    val_dir = os.path.join(split_base, 'validation')
    test_dir = os.path.join(split_base, 'test')

    if not os.path.exists(split_base):
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        os.makedirs(test_dir)
        class_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        if not class_names:
            raise ValueError(f"No class folders found in '{dataset_dir}'.")
        random.seed(42)
        for class_name in class_names:
            images = glob(os.path.join(dataset_dir, class_name, '*'))
            random.shuffle(images)
            n = len(images)
            n_train = int(0.7 * n)
            n_val = int(0.15 * n)
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

    # Prepare data
    train_loader, val_loader, class_names = prepare_data(split_base)
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    # Model
    model = AdvancedImageClassifierTorch(num_classes=num_classes, base_model_name='mobilenet_v2').to(device)
    print(model)

    # Train
    train_model(model, train_loader, val_loader)

    # Fine-tune (unfreeze base model)
    for param in model.base.parameters():
        param.requires_grad = True
    train_model(model, train_loader, val_loader, num_epochs=FINE_TUNE_EPOCHS, lr=FINE_TUNE_LR, save_path='best_finetuned_model.pt')

    # Evaluate
    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(os.path.join(split_base, 'test'), transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model.load_state_dict(torch.load('best_finetuned_model.pt', map_location=device))
    evaluate_model(model, test_loader, class_names)

if __name__ == "__main__":
    main()