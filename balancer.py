import pandas as pd

# Load CSV
df = pd.read_csv("train.csv")

# How many samples per class you want
samples_per_class = 1400

balanced_rows = []
for cls in range(5):
    cls_rows = df[df['label'] == cls]
    if len(cls_rows) > samples_per_class:
        cls_rows = cls_rows.sample(samples_per_class, random_state=42)
    balanced_rows.append(cls_rows)

balanced_df = pd.concat(balanced_rows).reset_index(drop=True)

# Save new CSV
balanced_df.to_csv("train_balanced.csv", index=False)
print("Balanced CSV created. Samples per class:")
print(balanced_df['label'].value_counts())