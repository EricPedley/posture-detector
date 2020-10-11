import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=(80,60),
    color_mode="grayscale",
    batch_size=16,
    shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(80,60),
    color_mode="grayscale",
    batch_size=16,
    shuffle=True
)
model = keras.Sequential()
# block 1
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(80, 60, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer=keras.optimizers.SGD(lr=0.001,momentum=0.9),loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_ds,epochs=20,validation_data=val_ds)