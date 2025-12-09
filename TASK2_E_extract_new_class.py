import pandas as pd
import random

from sklearn.model_selection import train_test_split

# Load your new CSV
df_new = pd.read_csv("new_train.csv")

# Check the unique classes
print(df_new['label'].unique())

# Filter for pinus_flexilis class (replace with the exact label in your CSV)
df_pinus = df_new[df_new['label'] == "pinus_flexilis"].copy()

# Randomly select 100 images
df_pinus_sample = df_pinus.sample(n=100, random_state=42)

# Rename the label to 5 (new class)
df_pinus_sample['label'] = 5

# Save the few-shot CSV for inclusion in training
df_pinus_sample.to_csv("new_class_fewshot.csv", index=False)

print(df_pinus_sample.head())
print(f"Number of samples: {len(df_pinus_sample)}")
df_new_train, df_new_test = train_test_split(df_pinus_sample, test_size=0.3, random_state=42)

# Save them
df_new_train.to_csv("new_class_train.csv", index=False)
df_new_test.to_csv("new_class_test.csv", index=False)