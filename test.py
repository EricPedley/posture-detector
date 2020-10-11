# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# import numpy as np
# x=np.array([0,1,2,3,4,5,6,7,8,9])
# y=np.array([1,-1,-3,-5,-7,-9,-11,-13,-15,-17])

# model = keras.Sequential([
#     layers.Dense(1)
# ])

# model.compile(optimizer = 'SGD', loss='mean_squared_error')

# model.fit(x,y,epochs=1000)

# print(model.predict([-3]))
import tensorflow as tf
from tensorflow import keras
from keras import layers

# datagen = ImageDataGenerator()
# train_it = datagen.flow_from_directory('data/train',class_mode='binary',color_mode="grayscale",target_size=(160,90))
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="training",
    seed=1337,
    color_mode="grayscale",
    image_size=(160,90),
    batch_size=16
)