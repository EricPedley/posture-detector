import argparse
import os
import numpy as np
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
from PIL import Image

from .fcrn import ResNet50UpProj
tf.disable_v2_behavior()
height = 228
width = 304
channels = 3
batch_size = 1

def init_net(model_path):
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    net = ResNet50UpProj({'data': input_node}, batch_size, 1, False)
    sess=tf.Session()
    saver = tf.train.Saver()     
    saver.restore(sess, model_path)
    return net,sess,input_node

def transform_img(img):
    img2 = img.resize([width,height], Image.ANTIALIAS)
    img2 = np.array(img2).astype('float32')
    return np.expand_dims(np.asarray(img2), axis = 0)

def predict(net,sess,input_node,img):
    pred = sess.run(net.get_output(), feed_dict={input_node: img})
    return pred

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    net,sess,input_node = init_net(args.model_path)
    pred = predict(sess,net,input_node,Image.open(args.image_paths))
    fig = plt.figure()
    ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
    fig.colorbar(ii)
    plt.show()
    os._exit(0)

if __name__ == '__main__':
    main()

        



