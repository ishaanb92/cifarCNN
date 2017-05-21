import numpy as np
import tensorflow as tf

# Some defines
NUM_EPOCHS_PER_DECAY = 100.0
LEARNING_RATE_DECAY_FACTOR = 0.5
INITIAL_LEARNING_RATE = 0.01
TRAINING_BATCH_SIZE = 128
TEST_BATCH_SIZE = 10000
MOVING_AVERAGES_DECAY = 0.9999
IMAGE_SIZE = 24

#Helper functions
def weights_initialize(shape,dev,name):
    initial = tf.truncated_normal(shape,stddev = dev)
    return tf.Variable(initial,name = name)

def bias_initialize(shape,name):
    initial = tf.constant(0.0,shape=shape)
    return tf.Variable(initial, name = name)

# Distorts training images to improve prediction accuracy
def distort_image(image):
    # Inputs:
        # A 4-D image tensor : Batch size x height x width  x channels
    # Returns:
        # A 4-D distorted image tensor
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    # Unpack 4-D tensor into a list of 3-D tensors to iterate over
    image_list = tf.unstack(image,axis=0)
    for img in image_list:
        img = tf.random_crop(img, [height,width, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img,max_delta=63)
        img = tf.image.random_contrast(img,lower = 0.2,upper = 1.8)
        img = tf.image.per_image_standardization(img)
        img.set_shape([height,width,3])
    # Set the shape
    distorted_image = tf.stack(image_list,axis=0)
    return distorted_image

# Centrally crops test set images to 24x24x3
def crop_test_image(image):
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    image_list = tf.unpack(image,axis=0)
    for img in image_list:
        img = tf.image.resize_image_with_crop_or_pad(img,height,width)
        img = tf.image.per_image_standardization(img)
        img.set_shape([height,width,3])
    cropped_image = tf.stack(image_list,axis=0)
    return cropped_image

def inference(image,training = True):
    # Re-shape the images
    if training:
        image_reshape = tf.reshape(image,[TRAINING_BATCH_SIZE,32,32,3])
    else:
        image_reshape = tf.reshape(image,[TEST_BATCH_SIZE,32,32,3])

    tf.summary.image('Images',image_reshape) # Adding visualization for image

    if training == True:
        # Distort Image
        input_image_tensor = distort_image(image_reshape)
    else:
        # Centrally crop image (to 24x24)
        input_image_tensor = crop_test_image(image_reshape)

    tf.summary.image('Images',input_image_tensor) # Adding visualization for image

    # 1st convolutional layer
    Wconv1 = weights_initialize([5,5,3,64],5e-2,"Wconv1")
    bconv1 = bias_initialize([64],"bconv1")
    conv1 = tf.nn.conv2d(input_image_tensor,Wconv1,[1,1,1,1],padding = 'SAME')
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
    if training:
        pool2_flat = tf.reshape(pool2,[TRAINING_BATCH_SIZE,pool2.get_shape()[1].value*pool2.get_shape()[2].value*64])
    else:
        pool2_flat = tf.reshape(pool2,[TEST_BATCH_SIZE,pool2.get_shape()[1].value*pool2.get_shape()[2].value*64])
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
    tf.summary.scalar('Training Loss',loss)
    return loss

# Training step computation
def create_train_step(loss,global_step,num_examples):
    num_batches_per_epoch = num_examples/TRAINING_BATCH_SIZE
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY),LEARNING_RATE_DECAY_FACTOR,staircase = True)
    tf.summary.scalar('Learning Rate',learning_rate)
    # Create op for maintaining moving average of weights and biases (trainable variables)
    variable_averages = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGES_DECAY)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # Define the minimization step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)

    # Defining the train_op
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name = 'train')
    return train_op

def evaluate(out,labels):
    correct_prediction = tf.nn.in_top_k(out,labels,1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('Correct Predications',accuracy)
    return accuracy

