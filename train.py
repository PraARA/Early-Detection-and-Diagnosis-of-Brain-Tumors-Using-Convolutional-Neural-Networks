import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import get_datasets

# Load datasets
train_ds, val_ds, test_ds = get_datasets()

# CNN model
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (yes/no)
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Evaluate
loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.2f}")

# Save model
model.save("brain_tumor_cnn.h5")
