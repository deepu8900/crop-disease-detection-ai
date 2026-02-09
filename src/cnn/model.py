import tensorflow as tf

def build_cnn_model(num_classes):

    model = tf.keras.Sequential([

        tf.keras.layers.Rescaling(1./255, input_shape=(224,224,3)),

        tf.keras.layers.Conv2D(32,3,activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64,3,activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128,3,activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(256,3,activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(num_classes,activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
