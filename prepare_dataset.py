"""
This script prepares the image dataset for helmet detection.
It organizes and processes images before they are used for model training.
"""
import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Dataset locations
RAW_IMAGES = Path(r"C:\Users\Ruchitha\Downloads\helmet_dataset_raw\images")
RAW_ANNOTATIONS = Path(r"C:\Users\Ruchitha\Downloads\helmet_dataset_raw\annotations")

OUTPUT = Path(r"C:\Users\Ruchitha\Face-Mask-Helmet-Detection\dataset")

# Create folders
for split in ["train", "validation", "test"]:
    for label in ["helmet", "no_helmet"]:
        (OUTPUT / split / label).mkdir(parents=True, exist_ok=True)

# Find all XML annotation files
xml_files = list(RAW_ANNOTATIONS.glob("*.xml"))

print(f"Found {len(xml_files)} annotation files.")

helmet_images = []
no_helmet_images = []

for xml_file in xml_files:
    try:
        root = ET.parse(xml_file).getroot()

        image_name = root.findtext("filename")

        if not image_name:
            continue

        image_path = RAW_IMAGES / image_name

        if not image_path.exists():
            # Try matching by filename stem
            possible = list(RAW_IMAGES.glob(xml_file.stem + ".*"))
            if possible:
                image_path = possible[0]
            else:
                continue

        labels = []

        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name:
                labels.append(name.lower().strip())

        # Detect helmet-related labels
        if any("helmet" in label and "no" not in label and "without" not in label
               for label in labels):
            helmet_images.append(image_path)

        elif any("no" in label or "without" in label for label in labels):
            no_helmet_images.append(image_path)

    except Exception as e:
        print(f"Error reading {xml_file.name}: {e}")

print(f"Helmet images: {len(helmet_images)}")
print(f"No-helmet images: {len(no_helmet_images)}")

# Combine and shuffle
random.seed(42)

all_data = (
    [(img, "helmet") for img in helmet_images] +
    [(img, "no_helmet") for img in no_helmet_images]
)

random.shuffle(all_data)

if len(all_data) == 0:
    print("\nERROR: No images were classified.")
    print("We need to inspect the XML label names.")
    exit()

# Split: 70% train, 15% validation, 15% test
total = len(all_data)
train_end = int(total * 0.70)
val_end = int(total * 0.85)

splits = {
    "train": all_data[:train_end],
    "validation": all_data[train_end:val_end],
    "test": all_data[val_end:]
}

# Copy images
for split_name, items in splits.items():
    for image_path, label in items:
        destination = OUTPUT / split_name / label / image_path.name
        shutil.copy2(image_path, destination)

    print(f"{split_name}: {len(items)} images")

print("\nDataset preparation completed!")