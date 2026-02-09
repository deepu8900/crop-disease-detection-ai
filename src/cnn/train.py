from dataset_loader import load_datasets
from model import build_cnn_model
import tensorflow as tf

DATA_PATH = "data/processed"
MODEL_SAVE_PATH = "models/cnn_crop_disease.h5"

train_ds, val_ds, test_ds, class_names = load_datasets(DATA_PATH)

print("Classes:", class_names)

model = build_cnn_model(len(class_names))

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

print("Evaluating on test set...")
model.evaluate(test_ds)
