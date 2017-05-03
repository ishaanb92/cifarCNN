import numpy as np
import tensorflow as tf

# Some defines
NUM_EPOCHS_PER_DECAY = 50.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.01
BATCH_SIZE = 128

#Helper functions
def weights_initialize(shape,dev,name):
    initial = tf.truncated_normal(shape,stddev = dev)
    return tf.Variable(initial,name = name)

def bias_initialize(shape,name):
    initial = tf.constant(0.0,shape=shape)
    return tf.Variable(initial, name = name)

# Construct the CNN architecture
def inference(image):
    # Re-shape the images
    image_reshape = tf.reshape(image,[-1,32,32,3])
    # 1st convolutional layer
    Wconv1 = weights_initialize([5,5,3,64],5e-2,"Wconv1")
    bconv1 = bias_initialize([64],"bconv1")
    conv1 = tf.nn.conv2d(image_reshape,Wconv1,[1,1,1,1],padding = 'SAME')
    layer_1 = tf.nn.relu(tf.nn.bias_add(conv1,bconv1))

    # Pooling
    pool1 = tf.nn.max_pool(layer_1,ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

    #Normalize
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # 2nd convolutional layer
    Wconv2 = weights_initialize([5,5,64,64],0.1,"Wconv2")
    bconv2 = bias_initialize([64],"bconv2")
    conv2 = tf.nn.conv2d(norm1,Wconv2,[1,1,1,1],padding = 'SAME')
    layer_2 = tf.nn.relu(tf.nn.bias_add(conv2,bconv2))

    #Normalize
    norm2  = tf.nn.lrn(layer_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # Pooling
    pool2 = tf.nn.max_pool(norm2,ksize = [1,3,3,1], strides = [1,2,2,1], padding= 'SAME')

    # FC 1  Layer
    W_fc1 = weights_initialize([pool2.get_shape()[1].value*pool2.get_shape()[2].value*64,384],0.04,"W_fc1") # 384 taken from original CIFAR classifier
    b_fc1 = bias_initialize([384],"b_fc1");
    pool2_flat = tf.reshape(pool2,[-1,pool2.get_shape()[1].value*pool2.get_shape()[2].value*64])
    fc_1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1)

    # FC 2 Layer
    W_fc2 = weights_initialize([384,192],0.004,"W_fc2") # Shape taken from original CIFAR classifier
    b_fc2 = bias_initialize([192],"b_fc2");
    fc_2 = tf.nn.relu(tf.matmul(fc_1, W_fc2) + b_fc2)

    # Output Layer
    W_out = weights_initialize([192,10],1/192.0,"W_out")
    b_out = bias_initialize([10],"b_out")
    # Not applied the non-linearity yet for the output. Softmax to model "inhibition", suppresses multiple activations
    out = tf.add(tf.matmul(fc_2, W_out),b_out) # Output is a 1-D vector with 10 elements ( = #classes)
    regularizer = tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(Wconv2) + tf.nn.l2_loss(Wconv1)
    return out,regularizer

# Cost Model
def loss(out,regularizer,labels):
    lmbda = tf.constant(0.1) # Determines rate of weight decay
    loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=out) + lmbda*regularizer)
    return loss

# Training step computation
def create_train_step(loss,global_step,num_examples):
    num_batches_per_epoch = num_examples/BATCH_SIZE
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY),LEARNING_RATE_DECAY_FACTOR,staircase = True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    return train_step

def evaluate(out,labels):
    correct_prediction = tf.nn.in_top_k(out,labels,1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    return accuracy

