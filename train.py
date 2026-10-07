import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# First training stage
EPOCHS = 20

# Fine-tuning stage
FINE_TUNE_EPOCHS = 10


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n======================================")
print("LOADING DATASET")
print("======================================\n")


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)


validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_ds.class_names

print("\n======================================")
print("CLASS NAMES")
print("======================================")

print(class_names)

print("\nClass mapping:")
print("0 = helmet")
print("1 = no_helmet")


# Safety check
if class_names != ["helmet", "no_helmet"]:
    raise ValueError(
        f"Unexpected class order: {class_names}\n"
        "Expected: ['helmet', 'no_helmet']"
    )


# ============================================================
# COUNT TRAINING IMAGES
# ============================================================

all_labels = []

for images, labels in train_ds:
    all_labels.extend(
        labels.numpy().flatten()
    )


all_labels = np.array(
    all_labels
).astype(int)


class_counts = np.bincount(
    all_labels
)


print("\n======================================")
print("CLASS COUNTS")
print("======================================")

for i, name in enumerate(class_names):
    print(
        f"{name}: {class_counts[i]} images"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_images = len(all_labels)

number_of_classes = len(class_counts)

class_weight = {}

for i in range(number_of_classes):

    class_weight[i] = (
        total_images
        /
        (
            number_of_classes
            *
            class_counts[i]
        )
    )


print("\n======================================")
print("CLASS WEIGHTS")
print("======================================")

for i, weight in class_weight.items():

    print(
        f"{class_names[i]}: {weight:.3f}"
    )


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

validation_ds = validation_ds.prefetch(
    AUTOTUNE
)

test_ds = test_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.08
    ),

    tf.keras.layers.RandomZoom(
        0.10
    ),

    tf.keras.layers.RandomContrast(
        0.10
    )
])


# ============================================================
# LOAD MOBILENETV2
# ============================================================

print("\n======================================")
print("LOADING MOBILENETV2")
print("======================================\n")


base_model = tf.keras.applications.MobileNetV2(

    input_shape=(
        224,
        224,
        3
    ),

    include_top=False,

    weights="imagenet"
)


# Freeze MobileNetV2 during first stage

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(
        224,
        224,
        3
    )
)


# Data augmentation

x = data_augmentation(
    inputs
)


# MobileNetV2 preprocessing

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)


# MobileNetV2 feature extraction

x = base_model(
    x,
    training=False
)


# Global average pooling

x = tf.keras.layers.GlobalAveragePooling2D()(
    x
)


# Dense layer

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(
    x
)


# Dropout

x = tf.keras.layers.Dropout(
    0.4
)(
    x
)


# Binary classification output

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(
    x
)


# Create final model

model = tf.keras.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE MODEL - STAGE 1
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n======================================")
print("MODEL SUMMARY")
print("======================================\n")

model.summary()


# ============================================================
# CALLBACKS - STAGE 1
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.000001
    ),

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "best_helmet_model.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
]


# ============================================================
# STAGE 1 - TRANSFER LEARNING
# ============================================================

print("\n======================================")
print("STAGE 1: TRANSFER LEARNING")
print("======================================\n")

print("MobileNetV2 layers are frozen.")
print("Training the classification layers...\n")


history = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    class_weight=class_weight,

    callbacks=callbacks
)


# ============================================================
# LOAD BEST STAGE 1 MODEL
# ============================================================

print("\n======================================")
print("LOADING BEST STAGE 1 MODEL")
print("======================================\n")


best_stage1_path = (
    MODEL_DIR /
    "best_helmet_model.keras"
)


model = tf.keras.models.load_model(
    best_stage1_path
)


# ============================================================
# FINE-TUNING MOBILENETV2
# ============================================================

print("\n======================================")
print("STAGE 2: FINE-TUNING MOBILENETV2")
print("======================================\n")


# Unfreeze MobileNetV2

base_model.trainable = True


# Freeze earlier layers
# Only last 30 layers will be trainable

for layer in base_model.layers[:-30]:
    layer.trainable = False


# Keep BatchNormalization layers frozen
# This makes fine-tuning more stable with a small dataset.

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False


print("Fine-tuning the last MobileNetV2 layers...")


# ============================================================
# RECOMPILE FOR FINE-TUNING
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# ============================================================
# FINE-TUNING CALLBACKS
# ============================================================

fine_tune_callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.000001
    ),

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "best_helmet_model.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
]


# ============================================================
# STAGE 2 - FINE-TUNING
# ============================================================

history_fine = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weight,

    callbacks=fine_tune_callbacks
)


# ============================================================
# LOAD FINAL BEST MODEL
# ============================================================

print("\n======================================")
print("LOADING FINAL BEST MODEL")
print("======================================\n")


best_model_path = (
    MODEL_DIR /
    "best_helmet_model.keras"
)


model = tf.keras.models.load_model(
    best_model_path
)


# ============================================================
# TEST MODEL
# ============================================================

print("\n======================================")
print("FINAL TEST RESULTS")
print("======================================\n")


test_results = model.evaluate(
    test_ds,
    verbose=1
)


# Print metrics clearly

for name, value in zip(
    model.metrics_names,
    test_results
):

    print(
        f"{name}: {value:.4f}"
    )


# ============================================================
# EXTRACT TEST ACCURACY
# ============================================================

test_accuracy = test_results[1]


print("\n--------------------------------------")
print(
    f"FINAL TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)
print("--------------------------------------")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR /
    "helmet_model.keras"
)


model.save(
    final_model_path
)


print("\n======================================")
print("FINAL MODEL SAVED")
print("======================================")

print(
    f"\n{final_model_path}"
)


# ============================================================
# CREATE ACCURACY GRAPH
# ============================================================

print("\n======================================")
print("CREATING ACCURACY GRAPH")
print("======================================\n")


# Accuracy from Stage 1

stage1_train_accuracy = (
    history.history["accuracy"]
)

stage1_validation_accuracy = (
    history.history["val_accuracy"]
)


# Accuracy from Stage 2

stage2_train_accuracy = (
    history_fine.history["accuracy"]
)

stage2_validation_accuracy = (
    history_fine.history["val_accuracy"]
)


# Combine both stages

train_accuracy = (
    stage1_train_accuracy
    +
    stage2_train_accuracy
)


validation_accuracy = (
    stage1_validation_accuracy
    +
    stage2_validation_accuracy
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    range(1, len(train_accuracy) + 1),
    train_accuracy,
    label="Training Accuracy"
)


plt.plot(
    range(1, len(validation_accuracy) + 1),
    validation_accuracy,
    label="Validation Accuracy"
)


# Mark beginning of fine-tuning

plt.axvline(
    x=len(stage1_train_accuracy),
    linestyle="--",
    label="Fine-Tuning Starts"
)


plt.title(
    "Training vs Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.legend()


plt.grid(
    True
)


# ============================================================
# SAVE GRAPH
# ============================================================

plot_path = (
    MODEL_DIR /
    "accuracy_plot.png"
)


plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    f"Accuracy graph saved to:\n{plot_path}"
)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n======================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("======================================")

print("\nGenerated files:")

print(
    "✓ models/helmet_model.keras"
)

print(
    "✓ models/best_helmet_model.keras"
)

print(
    "✓ models/accuracy_plot.png"
)

print("\nNext steps:")

print(
    "1. Run: python webcam.py"
)

print(
    "2. Run: streamlit run app.py"
)