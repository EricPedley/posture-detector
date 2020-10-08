import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
img = load_img('data\\bad posture frames\out1.png', color_mode="grayscale")
img_array = img_to_array(img)
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
# for i in range(0,90):
#     print(img_array.item(i))
print(img_array.shape)
print(train_images.shape)
#print(tf.__version__)