import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import xgboost as xgb

# Load pretrained MobileNetV3
from torchvision.models import MobileNet_V3_Small_Weights, EfficientNet_B0_Weights

start_time = time.time()
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)


# Remove classifier -> keep feature extractor
feature_extractor = nn.Sequential(*list(model.children())[:-1])  # all layers except classifier
feature_extractor.eval()  # set to eval mode

# Example dataset
csv_file = "train_split.csv"
img_dir = "train_images"

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

# transforms
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])

dataset = CassavaDataset(csv_file, img_dir, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Extract features
features = []
labels = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = feature_extractor.to(device)

with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        feats = feature_extractor(imgs)
        feats = feats.view(feats.size(0), -1)  # flatten to 2D
        features.append(feats.cpu().numpy())
        labels.append(lbls.numpy())

X = np.vstack(features)
y = np.hstack(labels)

print("Feature matrix shape:", X.shape)  # should now be (5687, 576)
print("Labels shape:", y.shape)
# Convert to arrays
X = np.vstack(features)
y = np.hstack(labels)

print("Feature matrix shape:", X.shape)  # [num_samples, feature_dim]
print("Labels shape:", y.shape)



# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
# Random Forest with limits to prevent overfitting
gfc =xgb.XGBClassifier(
    n_estimators=175,
    max_depth=4,
    learning_rate=0.02,
    tree_method="hist",   # <--- GPU!
    device="cuda"
)

gfc.fit(X_train, y_train)

train_acc = gfc.score(X_train, y_train)
print("Training Accuracy:", train_acc)

# Evaluate
y_pred = gfc.predict(X_val)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print("Validation Accuracy:", acc)
print("Confusion Matrix:\n", cm)

# Load test set
test_dataset = CassavaDataset("test_split.csv", img_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

features_test = []
labels_test = []

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        feats = feature_extractor(imgs)
        feats = feats.view(feats.size(0), -1)
        features_test.append(feats.cpu().numpy())
        labels_test.append(lbls.numpy())

X_test = np.vstack(features_test)
y_test = np.hstack(labels_test)

# Evaluate on your dedicated test set
y_pred = gfc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Test Accuracy:", acc)
print("Confusion Matrix:\n", cm)


end_time = time.time()

runtime_seconds = end_time - start_time
print(f"Runtime: {runtime_seconds:.2f} seconds")