from pathlib import Path
import tensorflow as tf

# Point to datasets/Training and datasets/Testing
base_dir = Path("datasets")
train_dir = base_dir / "Training"
test_dir = base_dir / "Testing"

IMG_SIZE = (180, 180)
BATCH_SIZE = 32

def get_datasets():
    # Training + validation split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Testing dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    return train_ds, val_ds, test_ds
