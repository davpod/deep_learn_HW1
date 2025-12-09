import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Hyperparameters
# -----------------------------
NUM_CLASSES = 5
INPUT_SIZE = 224
BATCH_SIZE = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
K_FOLDS = 5
DROPOUT_PROB = 0.3

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
# 3. Model
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

# -----------------------------
# 4. Test-time augmentation transforms
# -----------------------------
tta_transforms = [
    transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                        transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                        transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor()]),
]
test_transform_basic = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor()
])
# -----------------------------
# 5. Load test dataset
# -----------------------------
csv_file_test = "test_split.csv"
img_dir = "train_images"
test_dataset = CassavaDataset(csv_file_test, img_dir, transform=tta_transforms[0])  # placeholder transform
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
true_labels = test_dataset.df['label'].values

# Load dataset
test_dataset_basic = CassavaDataset("test_split.csv", img_dir, transform=test_transform_basic)
test_loader_basic = DataLoader(test_dataset_basic, batch_size=BATCH_SIZE, shuffle=False)

# True labels
true_labels_basic = test_dataset_basic.df['label'].values
# -----------------------------
# 6. Load trained fold models
# -----------------------------
loaded_models = []
for fold in range(1, K_FOLDS + 1):
    model = SmallCNN().to(DEVICE)
    model_path = f"2C_models/model_fold{fold}.pth"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    loaded_models.append(model)
print(f"Loaded {len(loaded_models)} fold models")

# -----------------------------
# 7. Run non TTA ensemble inference
# -----------------------------
ensemble_preds_basic = []

with torch.no_grad():
    for imgs, _ in test_loader_basic:
        imgs = imgs.to(DEVICE)

        # Average predictions across folds
        fold_outputs = []
        for model in loaded_models:
            model.eval()
            out = model(imgs)
            fold_outputs.append(F.softmax(out, dim=1))

        # Mean across folds
        avg_outputs = torch.mean(torch.stack(fold_outputs), dim=0)
        _, preds = torch.max(avg_outputs, 1)
        ensemble_preds_basic.extend(preds.cpu().numpy())
accuracy_basic = np.mean(np.array(ensemble_preds_basic) == np.array(true_labels_basic))
print(f"Ensemble Accuracy without TTA: {accuracy_basic:.4f}")
# -----------------------------
# 8. Run TTA ensemble inference
# -----------------------------
ensemble_preds = []

with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.cpu()  # temporarily move to CPU for manual TTA

        tta_outputs = []

        for tta_tf in tta_transforms:
            # Apply TTA transform
            aug_imgs = torch.stack([tta_tf(transforms.ToPILImage()(img)) for img in imgs])
            aug_imgs = aug_imgs.to(DEVICE)

            # Get predictions from each fold
            fold_outputs = []
            for model in loaded_models:
                out = model(aug_imgs)
                fold_outputs.append(F.softmax(out, dim=1))

            # Average over folds
            fold_mean = torch.mean(torch.stack(fold_outputs), dim=0)
            tta_outputs.append(fold_mean)

        # Average over TTA transforms
        final_output = torch.mean(torch.stack(tta_outputs), dim=0)
        _, preds = torch.max(final_output, 1)
        ensemble_preds.extend(preds.cpu().numpy())

# -----------------------------
# 8. Compute TTA accuracy
# -----------------------------
accuracy_tta = np.mean(np.array(ensemble_preds) == np.array(true_labels))
print(f"TTA Ensemble Accuracy: {accuracy_tta:.4f}")

improvement = accuracy_tta - accuracy_basic
print(f"Accuracy improvement with TTA: {improvement:.4f}")
# -----------------------------
# 9. Confusion matrix
# -----------------------------
cm = confusion_matrix(true_labels, ensemble_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
            yticklabels=[f"Class {i}" for i in range(NUM_CLASSES)])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("TTA Ensemble Confusion Matrix")
plt.show()

cm = confusion_matrix(true_labels_basic, ensemble_preds_basic)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
            yticklabels=[f"Class {i}" for i in range(NUM_CLASSES)])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("NON-TTA Ensemble Confusion Matrix")
plt.show()
