import numpy as np
import tensorflow as tf
import cifar10_input
import os

# Some defines
NUM_EPOCHS_PER_DECAY = 350.0/6
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.3
TRAINING_BATCH_SIZE = 128
NUM_TEST_EXAMPLES = 10000
MOVING_AVERAGES_DECAY = 0.9999
IMAGE_SIZE = 24
NUM_TRAINING_EXAMPLES = 50000

#Helper functions
def weights_initialize(shape,dev,decay,name):
    var = tf.get_variable(name,shape,initializer = tf.truncated_normal_initializer(stddev=dev,dtype=tf.float32),dtype=tf.float32)
    if decay != 0:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),decay,name = 'weight_decay')
        tf.add_to_collection('losses',weight_decay)
    return var

def bias_initialize(shape,name):
    var = tf.get_variable(name,shape,initializer=tf.constant_initializer(0.0))
    return var

# Input handler functions
# Returns 4-D image tensor : [BATCH_SIZE,h,w,channels]
#         1-D label tensor : [BATCH_SIZE]
def distorted_inputs():
    data_dir = os.path.join(os.getcwd(),'cifar-10-batches-bin')
    images,labels = cifar10_input.distorted_inputs(data_dir = data_dir, batch_size = TRAINING_BATCH_SIZE)
    return images,labels

def inputs(eval_data):
    data_dir = os.path.join(os.getcwd(),'cifar-10-batches-bin')
    images,labels = cifar10_input.inputs(eval_data = eval_data,
                                         data_dir = data_dir,
                                         batch_size = TRAINING_BATCH_SIZE)
    return images,labels


def inference(image):

    # 1st convolutional layer
    Wconv1 = weights_initialize([5,5,3,64],5e-2,0.0,"Wconv1")
    bconv1 = bias_initialize([64],"bconv1")
    conv1 = tf.nn.conv2d(image,Wconv1,[1,1,1,1],padding = 'SAME')
    # gamma : Scale parameter
    # beta : Shift parameter
    # These trainable parameters are introduced so that the model while reducing internal covariate shift
    # does not lose it's representational power (Names are consistent with notation in the BN paper [2015])
    gamma_conv1 = tf.get_variable("gamma_conv1",shape=[64],initializer=tf.constant_initializer(1.0))
    beta_conv1 = tf.get_variable("beta_conv1",shape=[64],initializer=tf.constant_initializer(0.0))
    # Apply BN
    batch_mean_conv1, batch_var_conv1 = tf.nn.moments(conv1,[0])
    conv1_BN = tf.nn.batch_normalization(conv1,batch_mean_conv1,batch_var_conv1,beta_conv1,gamma_conv1,variance_epsilon=1e-3)
    layer_1 = tf.nn.relu(tf.nn.bias_add(conv1_BN,bconv1))

    # Pooling
    pool1 = tf.nn.max_pool(layer_1,ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

    #Normalize
    #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # 2nd convolutional layer
    Wconv2 = weights_initialize([5,5,64,64],0.1,0.0,"Wconv2")
    bconv2 = bias_initialize([64],"bconv2")
    conv2 = tf.nn.conv2d(pool1,Wconv2,[1,1,1,1],padding = 'SAME')
    # Add BN here
    gamma_conv2 = tf.get_variable("gamma_conv2",shape=[64],initializer=tf.constant_initializer(1.0))
    beta_conv2 = tf.get_variable("beta_conv2",shape=[64],initializer=tf.constant_initializer(0.0))
    # Apply BN
    batch_mean_conv2, batch_var_conv2 = tf.nn.moments(conv2,[0])
    conv2_BN = tf.nn.batch_normalization(conv2,batch_mean_conv2,batch_var_conv2,beta_conv2,gamma_conv2,variance_epsilon=1e-3)
    layer_2 = tf.nn.relu(tf.nn.bias_add(conv2_BN,bconv2))

    #Normalize
    #norm2  = tf.nn.lrn(layer_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # Pooling
    pool2 = tf.nn.max_pool(layer_2,ksize = [1,3,3,1], strides = [1,2,2,1], padding= 'SAME')

    # FC 1  Layer
    W_fc1 = weights_initialize([pool2.get_shape()[1].value*pool2.get_shape()[2].value*64,384],0.04,0.004/5,"W_fc1") # 384 taken from original CIFAR classifier
    b_fc1 = bias_initialize([384],"b_fc1");
    pool2_flat = tf.reshape(pool2,[TRAINING_BATCH_SIZE,pool2.get_shape()[1].value*pool2.get_shape()[2].value*64])
    pool2_mul = tf.matmul(pool2_flat, W_fc1)
    gamma_pool2 = tf.get_variable("gamma_pool1",shape=[384],initializer=tf.constant_initializer(1.0))
    beta_pool2 = tf.get_variable("beta_pool2",shape=[384],initializer=tf.constant_initializer(0.0))
    # Apply BN
    batch_mean_pool2, batch_var_pool2 = tf.nn.moments(pool2_mul,[0])
    pool2_BN = tf.nn.batch_normalization(pool2_mul,batch_mean_pool2,batch_var_pool2,beta_pool2,gamma_pool2,variance_epsilon=1e-3)
    fc_1 = tf.nn.relu(tf.nn.bias_add(pool2_BN,b_fc1))

    # FC 2 Layer
    W_fc2 = weights_initialize([384,192],0.004,0.004/5,"W_fc2") # Shape taken from original CIFAR classifier
    b_fc2 = bias_initialize([192],"b_fc2");
    fc2_mul = tf.matmul(fc_1, W_fc2)
    # Add BN here
    gamma_fc2 = tf.get_variable("gamma_fc2",shape=[192],initializer=tf.constant_initializer(1.0))
    beta_fc2 = tf.get_variable("beta_fc2",shape=[192],initializer=tf.constant_initializer(0.0))
    # Apply BN
    batch_mean_fc2, batch_var_fc2 = tf.nn.moments(fc2_mul,[0])
    pool2_BN = tf.nn.batch_normalization(fc2_mul,batch_mean_fc2,batch_var_fc2,beta_fc2,gamma_fc2,variance_epsilon=1e-3)
    fc_2 = tf.nn.relu(tf.nn.bias_add(pool2_BN,b_fc2))

    # Output Layer
    W_out = weights_initialize([192,10],1/192.0,0.0,"W_out")
    b_out = bias_initialize([10],"b_out")
    # Not applied the non-linearity yet for the output. Softmax to model "inhibition", suppresses multiple activations
    out = tf.add(tf.matmul(fc_2, W_out),b_out) # Output is a 1-D vector with 10 elements ( = #classes)
    return out

# Cost Model
def loss(out,labels):
    loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=out),name = 'cross_entropy')
    tf.add_to_collection('losses',loss)
    # Add L2 Loss terms to the cross entropy loss
    total_loss = tf.add_n(tf.get_collection('losses'),name='total_loss')
    tf.summary.scalar('Training Loss',total_loss)
    return total_loss

# Training step computation
def create_train_step(loss,global_step):
    num_batches_per_epoch = NUM_TRAINING_EXAMPLES/TRAINING_BATCH_SIZE
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY),LEARNING_RATE_DECAY_FACTOR,staircase = True)
    tf.summary.scalar('Learning Rate',learning_rate)
    # Define the minimization step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    # Defining the train_op
    with tf.control_dependencies([train_step]):
        train_op = tf.no_op(name = 'train')
    return train_op

def evaluate(out,labels):
    correct_prediction = tf.nn.in_top_k(out,labels,1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('Correct Predications',accuracy)
    return tf.cast(correct_prediction,tf.float32)

