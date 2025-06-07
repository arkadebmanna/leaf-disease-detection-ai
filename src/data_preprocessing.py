import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# File paths
csv_path = "data/PlantVillage_metadata.csv"
image_dir = "data/sample_images/"

# Load CSV file
df = pd.read_csv(csv_path)

# ✅ Check for missing image files
missing_files = [f for f in df["filename"] if not os.path.exists(os.path.join(image_dir, f))]
print(f"Missing image files: {missing_files}")

# ✅ Print total dataset size
print(f"Total images found: {len(df)}")

# ✅ Handle Small Dataset Issue (Remove Stratify)
if len(df) < 5:  # If dataset is small, avoid stratification
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
else:
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# ✅ Define ImageDataGenerator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# ✅ Create training generator
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col="filename",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# ✅ Create validation generator
val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=image_dir,
    x_col="filename",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# ✅ Debugging checks
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Validation filenames: {val_generator.filenames}")
