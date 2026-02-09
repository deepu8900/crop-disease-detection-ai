import tensorflow as tf

IMG_SIZE = (224,224)
BATCH_SIZE = 32

def load_datasets(DATA_PATH):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "/val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
