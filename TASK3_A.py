import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms, models
import pandas as pd
import random
import os
from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import ResNet18_Weights, AlexNet_Weights, MobileNet_V3_Small_Weights, EfficientNet_B0_Weights
import time


start_time = time.time()
# -----------------------------
# 1. Hyperparameters
# -----------------------------
NUM_CLASSES = 5
INPUT_SIZE = 224
BATCH_SIZE = 256
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
TRAIN_SIZE = 6000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -----------------------------
# 2. Dataset
# -----------------------------
class CassavaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        label = self.df.iloc[idx]['label']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------------
# 3. Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor()
])

# -----------------------------
# 4. Load dataset
# -----------------------------
csv_file = "train_split.csv"
img_dir = "train_images"
full_dataset = CassavaDataset(csv_file, img_dir, transform=train_transform)

# Subset for faster experimentation
all_indices = list(range(len(full_dataset)))
random.shuffle(all_indices)
subset_indices = all_indices[:TRAIN_SIZE]
subset_dataset = Subset(full_dataset, subset_indices)

# Split into train/val
val_ratio = 0.2
val_size = int(len(subset_dataset) * val_ratio)
train_size = len(subset_dataset) - val_size
train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# 5. Load pretrained models and modify last layer
# -----------------------------
def get_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        last_layer_name = "fc"
    elif model_name == "alexnet":
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        last_layer_name = "classifier.6"
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
        last_layer_name = "classifier.3"
    elif model_name == "efficientnet":
        model =  models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        last_layer_name = "classifier.1"
    else:
        raise ValueError("Unknown model name")
    return model, last_layer_name


def freeze_model(model, last_layer_name=None):
    #print(model.classifier)
    for param in model.parameters():
        param.requires_grad = False
    if last_layer_name is not None:
        if "classifier" in last_layer_name:
            idx = int(last_layer_name.split('.')[-1])
            layer = model.classifier[idx]
            for param in layer.parameters():
                param.requires_grad = True
        elif last_layer_name == "fc":
            for param in model.fc.parameters():
                param.requires_grad = True
    return model


model_name = "efficientnet"

model, last_layer_name = get_model(model_name, NUM_CLASSES)
model = freeze_model(model, last_layer_name)
model = model.to(DEVICE)

# -----------------------------
# 6. Training function
# -----------------------------
def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss, running_corrects = 0.0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == lbls.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)


        # Validation
        model.eval()
        val_loss_total = 0.0
        val_corrects = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss_total += loss.item() * imgs.size(0)  # sum over batch
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == lbls.data)

        # Average validation loss over all samples
        val_loss = val_loss_total / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    return model

# -----------------------------
# 7. Train model
# -----------------------------
trained_model = train_model(model, train_loader, val_loader)

# -----------------------------
# 8. Optional: Evaluate on separate test set
# -----------------------------
test_dataset = CassavaDataset("test_split.csv", img_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.CrossEntropyLoss()
true_labels = []
all_preds = []
test_loss_accum = 0
total_samples = 0
unique_correct = 0
unique_errors = 0

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        outputs = trained_model(imgs)

        # Compute loss
        loss = criterion(outputs, lbls)
        test_loss_accum += loss.item() * imgs.size(0)

        # Compute predictions
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        true_labels.extend(lbls.cpu().numpy())

        # Count correct/incorrect
        correct_mask = (preds == lbls)
        unique_correct += correct_mask.sum().item()
        unique_errors += (~correct_mask).sum().item()

        total_samples += lbls.size(0)

test_loss = test_loss_accum / total_samples
test_accuracy = unique_correct / total_samples

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Unique correct samples: {unique_correct}")
print(f"Unique errors: {unique_errors}")

# -----------------------------
# 9. Confusion matrix for ensemble
# -----------------------------
cm = confusion_matrix(true_labels, all_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
            yticklabels=[f"Class {i}" for i in range(NUM_CLASSES)])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.show()

end_time = time.time()

runtime_seconds = end_time - start_time
print(f"Runtime: {runtime_seconds:.2f} seconds")