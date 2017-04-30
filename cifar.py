import numpy as np
import tensorflow as tf
import data_helpers # Helper functions to fetch CIFAR data

# Some defines
MAX_STEPS = 125*1000 # 75k epochs
BATCH_SIZE = 256
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.01

#Helper functions
def weights_initialize(shape,dev):
    initial = tf.truncated_normal(shape,stddev = dev)
    return tf.Variable(initial)

def bias_initialize(shape):
    initial = tf.constant(0.0,shape=shape)
    return tf.Variable(initial)

cifar_dataset = data_helpers.load_data() # Loads training images/labels + test images/labels

# Inputs and labels
image = tf.placeholder(tf.float32,[None,3072])
labels = tf.placeholder(tf.int64,[None])



# Re-shape the images
image_reshape = tf.reshape(image,[-1,32,32,3])

# Construct the CNN architecture

# 1st convolutional layer
Wconv1 = weights_initialize([5,5,3,64],5e-2)
bconv1 = bias_initialize([64])
conv1 = tf.nn.conv2d(image_reshape,Wconv1,[1,1,1,1],padding = 'SAME')
layer_1 = tf.nn.relu(tf.nn.bias_add(conv1,bconv1))

# Pooling
pool1 = tf.nn.max_pool(layer_1,ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

#Normalize
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 2nd convolutional layer
Wconv2 = weights_initialize([5,5,64,64],0.1)
bconv2 = bias_initialize([64])
conv2 = tf.nn.conv2d(norm1,Wconv2,[1,1,1,1],padding = 'SAME')
layer_2 = tf.nn.relu(tf.nn.bias_add(conv2,bconv2))

#Normalize
norm2  = tf.nn.lrn(layer_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# Pooling
pool2 = tf.nn.max_pool(norm2,ksize = [1,3,3,1], strides = [1,2,2,1], padding= 'SAME')

# FC 1  Layer
W_fc1 = weights_initialize([pool2.get_shape()[1].value*pool2.get_shape()[2].value*64,384],0.04) # 384 taken from original CIFAR classifier
b_fc1 = bias_initialize([384]);
pool2_flat = tf.reshape(pool2,[-1,pool2.get_shape()[1].value*pool2.get_shape()[2].value*64])
fc_1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1)

# FC 2 Layer
W_fc2 = weights_initialize([384,192],0.004) # Shape taken from original CIFAR classifier
b_fc2 = bias_initialize([192]);
fc_2 = tf.nn.relu(tf.matmul(fc_1, W_fc2) + b_fc2)

# Output Layer
W_out = weights_initialize([192,10],1/192.0)
b_out = bias_initialize([10])
# Not applied the non-linearity yet for the output. Softmax to model "inhibition", suppresses multiple activations
out = tf.add(tf.matmul(fc_2, W_out),b_out) # Output is a 1-D vector with 10 elements ( = #classes)

# Cost Model
regularizer = tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(Wconv2) + tf.nn.l2_loss(Wconv1)
lmbda = tf.constant(0.1) # Determines rate of weight decay
loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=out) + lmbda*regularizer)

# Training step computation
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,int(BATCH_SIZE*NUM_EPOCHS_PER_DECAY),LEARNING_RATE_DECAY_FACTOR,staircase = True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)

# Evaluating the training
correct_prediction = tf.equal(tf.argmax(out, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init);


print ("Input size : ")
print (image_reshape.get_shape())
print("\n")

print ("Conv1 size: ")
print (conv1.get_shape())
print("\n")

print("Pool1 size: ")
print(pool1.get_shape())
print("\n")


print ("Conv2 size: ")
print (conv2.get_shape())
print("\n")

print("Pool2 size: ")
print(pool2.get_shape())
print("\n")

print("Pool2 reshape size: ")
print(pool2_flat.get_shape())
print("\n")

print ("FC 1 size: ")
print(fc_1.get_shape())
print("\n")

print ("FC 2 size: ")
print(fc_2.get_shape())
print("\n")

print ("Output layer size: ")
print(out.get_shape())
print("\n")

# Training
for i in range(MAX_STEPS):
    # Generate batch
    batch = np.random.choice(cifar_dataset['images_train'].shape[0], BATCH_SIZE)
    images_batch = cifar_dataset['images_train'][batch]
    labels_batch = cifar_dataset['labels_train'][batch]
    # Execute the train step
    sess.run(train_step,feed_dict = {image:images_batch,labels:labels_batch})
    if i%100 == 0:
        train_accuracy = sess.run(accuracy,feed_dict = {image:images_batch,labels:labels_batch})
        print ('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
# Testing
test_accuracy = sess.run(accuracy, feed_dict={ image: cifar_dataset['images_test'],labels: cifar_dataset['labels_test']})
print('Test accuracy {:g}'.format(test_accuracy))
