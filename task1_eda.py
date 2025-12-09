import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

# load csv
df = pd.read_csv("train.csv")

# Number of samples
num_samples = len(df)
print(f"Total number of samples: {num_samples}")

# List all images to double-check
image_folder = "train_images"
all_images = os.listdir(image_folder)
print(f"Number of image files: {len(all_images)}")
#21397 images

# Open a random image to inspect
sample_img = Image.open(os.path.join(image_folder, df.iloc[0]['image_id']))
print(f"Image size (width x height): {sample_img.size}")
#800x600
print(f"Image mode (channels): {sample_img.mode}")
#rgb=3 channels

# Number of classes
num_classes = df['label'].nunique()
print(f"Number of classes: {num_classes}")
#5 classes

#we have a large enough data set, and upon manual inspection of the images,it doesn't seem like
#augmantation will be needed as we posses images in all oriantations, some with very
#noisy background and some are more clear.
#although due to time and hardware limitation i will be cutting the train size to only 5k images
#so i will add augmentations mainly in the form of mirroring and maybe rotations
#but i will need to preprocess the data as its size is 800x600, which is quite huge, so i will resize
#it to 224x224 to save both time and to allow better integration in task 3 with other feature extractors
#since many older models that were trained on imagenet use 224x224

class_counts = Counter(df['label'])
print("Number of examples per class:")
for cls, count in class_counts.items():
    print(f"Class {cls}: {count}")

# Bar chart
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title("Class distribution")
plt.show()
#the data is heavily imbalanced, with around 60% of the images
#being of class 3
#we have 13k of class 3, 2k of class 1,2,4 each, and 1 k of class 0

#there are benchmarks on https://www.kaggle.com/competitions/cassava-leaf-disease-classification/leaderboard
#the top of the leader board are around 90% accuracy while 1'st place has 91%
#first place uses a mean of VIT amd resnext50, and then summs with efficientNet and mobilenet
#all of whom are pretrained imagenet models
#almost if not all other comoetitors also relied on pretrained models


# Show 3 samples per class
plt.figure(figsize=(15,10))

sample_imgs = df[df['label']==0]['image_id'].values[:3]

plt.figure(figsize=(15, 5))  # make the whole figure wide

for i, img_name in enumerate(sample_imgs):
    img = Image.open(os.path.join(image_folder, img_name))

    plt.subplot(1, 3, i + 1)   # 1 row, 3 columns
    plt.imshow(img)
    plt.axis('off')
    if i == 1:
        plt.title("Class 0")

plt.tight_layout()
plt.show()