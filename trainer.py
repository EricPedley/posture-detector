import tensorflow as tf
from tensorflow import keras
from model_example import make_model
model = make_model(input_shape=(80,60) + (1,), num_classes=2)
epochs = 25
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=(80,60),
    color_mode="grayscale",
    batch_size=8,
    shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(80,60),
    color_mode="grayscale",
    batch_size=8,
    shuffle=True
)
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
