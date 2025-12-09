import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train_balanced.csv")

# Number of images to put in test
TEST_SIZE = 1000

# Get label column for stratification
labels = df['label']

# Compute fraction for test size
test_fraction = TEST_SIZE / len(df)

train_df, test_df = train_test_split(
    df,
    test_size=test_fraction,
    stratify=labels,
    random_state=42
)

print("Total samples:", len(df))
print("Train samples:", len(train_df))
print("Test samples:", len(test_df))

train_df.to_csv("train_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)