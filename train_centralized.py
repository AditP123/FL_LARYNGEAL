import tensorflow as tf
import os
from prepare_data import load_data_for_fold, DATASET_PATH # Import from our previous script

# --- Configuration ---
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

# --- 1. Load and Combine Data from All Folds ---
print("Loading data from all folds...")
# Load data from each fold
train_ds_1, val_ds_1, num_classes = load_data_for_fold(os.path.join(DATASET_PATH, 'FOLD 1'))
train_ds_2, val_ds_2, _ = load_data_for_fold(os.path.join(DATASET_PATH, 'FOLD 2'))
train_ds_3, val_ds_3, _ = load_data_for_fold(os.path.join(DATASET_PATH, 'FOLD 3'))

# Concatenate them into a single, large dataset
full_train_ds = train_ds_1.concatenate(train_ds_2).concatenate(train_ds_3)
full_val_ds = val_ds_1.concatenate(val_ds_2).concatenate(val_ds_3)

# Shuffle the combined datasets for good measure
full_train_ds = full_train_ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
full_val_ds = full_val_ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)

print(f"\n✅ All data loaded and combined. Total number of classes: {num_classes}")

# --- 2. Build the Model using Transfer Learning (EfficientNetB0) ---
def build_model(num_classes):
    # Load the pre-trained EfficientNetB0 model, without its top classification layer
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(160, 160, 3),
        include_top=False, # We'll add our own top layer
        weights='imagenet' # Use weights pre-trained on ImageNet
    )
    
    # Freeze the base model layers to prevent them from being re-trained
    base_model.trainable = False
    
    # Create the new model on top
    inputs = tf.keras.Input(shape=(160, 160, 3))
    # EfficientNetB0 expects pixel values in [0, 255]. Our data is already in that range.
    # We apply the data augmentation we defined earlier
    x = base_model(inputs, training=False) # Set training=False for the frozen base
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x) # Regularization
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

print("Building the centralized model...")
model = build_model(num_classes)

# --- 3. Compile the Model ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model built and compiled successfully.")
model.summary()

# --- 4. Train the Model ---
print("\nStarting centralized training...")
history = model.fit(
    full_train_ds,
    epochs=NUM_EPOCHS,
    validation_data=full_val_ds
)

# --- 5. Evaluate the Final Model ---
print("\nCentralized training finished.")
final_loss, final_accuracy = model.evaluate(full_val_ds)
print(f"\nBenchmark Model - Final Validation Accuracy: {final_accuracy * 100:.2f}%")