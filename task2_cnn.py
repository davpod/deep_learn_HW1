import os
import random
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler,WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import OneCycleLR

# -----------------------------
# 1. Hyperparameters
# -----------------------------
NUM_CLASSES = 5
INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
K_FOLDS = 5
DROPOUT_PROB = 0.3
TRAIN_SIZE = 2000
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
# 3. Transforms (plain)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor()
])
# -----------------------------
# 4. Model
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_prob=DROPOUT_PROB):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256*(INPUT_SIZE//16)*(INPUT_SIZE//16), 256)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

#train
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    correct, total = 0, 0
    loss_accum = 0

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        loss_accum += loss.item() * imgs.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)

    return loss_accum / total, correct / total

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    loss_accum = 0
    outputs_all, labels_all = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            loss_accum += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

            outputs_all.append(F.softmax(outputs, dim=1).cpu())
            labels_all.append(lbls.cpu())

    return (
        loss_accum / total,
        correct / total,
        torch.cat(outputs_all),
        torch.cat(labels_all)
    )

# -----------------------------
# 5. Load dataset & subset
# -----------------------------
csv_file = "train_split.csv"
img_dir = "train_images"
full_dataset = CassavaDataset(csv_file, img_dir, transform=transform)

all_indices = list(range(len(full_dataset)))
random.shuffle(all_indices)
subset_indices = all_indices[:TRAIN_SIZE]
subset_dataset = Subset(full_dataset, subset_indices)

# Extract labels directly from the subset
subset_labels = [subset_dataset[i][1] for i in range(len(subset_dataset))]

# Count occurrences per class
counts = pd.Series(subset_labels).value_counts().sort_index()

print("Class counts in subset:")
print(counts)

test_dataset = CassavaDataset("test_split.csv", img_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

true_labels = test_dataset.df['label'].values
# -----------------------------
# 6. KFold cross-validation
# -----------------------------
trainval_idx = list(range(len(subset_dataset)))

labels = [subset_dataset[i][1] for i in trainval_idx]
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

fold_metrics = []
os.makedirs("saved_models", exist_ok=True)
fold_train_acc = []
fold_val_acc   = []
fold_train_loss = []
fold_val_loss   = []


for fold, (train_idx, val_idx) in enumerate(kfold.split(trainval_idx, labels)):
    print(f"\n--- Fold {fold+1} ---")
    # Adjust indices for subset_dataset
    train_idx = [trainval_idx[i] for i in train_idx]
    val_idx = [trainval_idx[i] for i in val_idx]

    train_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))

    model = SmallCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    steps_per_epoch = len(train_loader)


    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )

        val_loss, val_acc, _, _ = validate_one_epoch(
            model, val_loader, criterion, DEVICE
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

    fold_train_acc.append(train_accs[-1])
    fold_val_acc.append(val_accs[-1])
    fold_train_loss.append(train_losses[-1])
    fold_val_loss.append(val_losses[-1])

    # Save the model after training this fold
    model_path = f"saved_models/model_fold{fold + 1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model for fold {fold + 1} at {model_path}")

    fold_metrics.append({
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "model": model,
        "val_idx": val_idx
    })

#generate fold accuracy graph
plt.figure(figsize=(10,5))
plt.plot(fold_train_acc, label="Train Acc")
plt.plot(fold_val_acc, label="Val Acc")
plt.title("5-Fold Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(fold_train_loss, label="Train Loss")
plt.plot(fold_val_loss, label="Val Loss")
plt.title("5-Fold Loss")
plt.xlabel("Fold")
plt.ylabel("Loss")
plt.legend()
plt.show()
# -----------------------------
# 7. Ensemble predictions on test set
# -----------------------------
ensemble_preds = []
print("entering testing")
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(DEVICE)

        fold_outputs = []
        for fold_data in fold_metrics:
            model = fold_data["model"]
            model.eval()
            outputs = model(imgs)
            fold_outputs.append(F.softmax(outputs, dim=1))

        avg_outputs = torch.mean(torch.stack(fold_outputs), dim=0)
        _, preds = torch.max(avg_outputs, 1)
        ensemble_preds.extend(preds.cpu().numpy())

accuracy = np.mean(np.array(ensemble_preds) == np.array(true_labels))
print(f"\nEnsemble Test Accuracy: {accuracy:.4f}")

# -----------------------------
# 8. Visualize train/val curves for fold 1
# -----------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(fold_metrics[0]["train_loss"], label="Train Loss")
plt.plot(fold_metrics[0]["val_loss"], label="Val Loss")
plt.title("Fold 1 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(fold_metrics[0]["train_acc"], label="Train Acc")
plt.plot(fold_metrics[0]["val_acc"], label="Val Acc")
plt.title("Fold 1 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# -----------------------------
# 9. Confusion matrix for ensemble
# -----------------------------
cm = confusion_matrix(true_labels, ensemble_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
            yticklabels=[f"Class {i}" for i in range(NUM_CLASSES)])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Ensemble Confusion Matrix")
plt.show()

# -----------------------------
# 10. Show examples of high-confidence correct/incorrect predictions
# -----------------------------
test_iter = iter(test_loader)
imgs, lbls = next(test_iter)
imgs = imgs.to(DEVICE)
with torch.no_grad():
    fold_outputs = [F.softmax(fold_data["model"](imgs), dim=1) for fold_data in fold_metrics]
avg_outputs = torch.mean(torch.stack(fold_outputs), dim=0)
probs, preds = torch.max(avg_outputs, 1)

probs = probs.cpu().numpy()
preds = preds.cpu().numpy()
lbls = lbls.numpy()

print("\nHigh-confidence correct predictions:")
for i in np.argsort(-probs):
    if preds[i] == lbls[i]:
        plt.imshow(imgs[i].cpu().permute(1,2,0))
        plt.title(f"Pred: {preds[i]}, True: {lbls[i]}, Prob: {probs[i]:.2f}")
        plt.show()
        break

print("\nHigh-confidence incorrect predictions:")
for i in np.argsort(-probs):
    if preds[i] != lbls[i]:
        plt.imshow(imgs[i].cpu().permute(1,2,0))
        plt.title(f"Pred: {preds[i]}, True: {lbls[i]}, Prob: {probs[i]:.2f}")
        plt.show()
        break
