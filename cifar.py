import numpy as np
import tensorflow as tf
import data_helpers # Helper functions to fetch CIFAR data


#Helper functions
def weights_initialize(shape):
    initial = tf.truncate_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_initialize(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

cifar_dataset = data_helpers.load_data() # Loads training images/labels + test images/labels

# Inputs and labels
image = tf.placeholder(tf.float32,[None,3072])
label = tf.placeholder(tf.int32,[None])

# Re-shape the images
image_reshape = image.reshape(images,[-1,32,32,3])

# 1st convolutional layer
Wconv1 = weights_initialize([5,5,3,64])
bconv1 = bias_initialize([64])
conv1 = tf.nn.conv2d(image_reshape,Wconv1,[1,1,1,1],padding = 'SAME')
layer_1 = tf.nn.relu(tf.nn.bias_add(conv1,bconv1))
# Pooling
pool1 = tf.nn.max_pool(layer_1,ksize = [1,2,2,1], strides = [1,2,2,1]<Plug>PeepOpenadding = 'SAME')
#Normalize
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 2nd convolutional layer
Wconv2 = weights_initialize([5,5,64,64])
bconv2 = bias_initialize([64])
conv2 = tf.nn.conv2d(norm1,Wconv2,[1,1,1,1],padding = 'SAME')
layer_2 = tf.nn.relu(tf.nn.bias_add(conv2,bconv2))
#Normalize
norm2  = tf.nn.lrn(layer_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
# Pooling
pool2 = tf.nn.max_pool(norm2,ksize = [1,2,2,1], strides = [1,2,2,1]<Plug>PeepOpenadding = 'SAME')

