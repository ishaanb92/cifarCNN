import tensorflow as tf
import numpy as np
import cifar # Contains inference(),loss(),training()
import data_helpers # Helper functions to fetch CIFAR data
import os

MAX_STEPS = 125*1000 # 75k epochs
BATCH_SIZE = 128

def evaluate_batch(sess,accuracy,images_batch,label_batch,image_pl,label_pl):
    # Inputs:
        # sess : Current session
        # accuracy : op defined for computing the accuracy
        # images_batch : Tensor containing images in the current batch
        # label_batch : Tensor containing label in the current batch
        # image_pl : Placeholder tensor for images
        # label_pl : Placeholder tensor for label
    # Returns:
        # train_accuracy : fraction of correctly predicated label

    train_accuracy = sess.run(accuracy,feed_dict = {image_pl: images_batch,label_pl:label_batch})
    return train_accuracy



# Top level training function
def run_training():

    with tf.Graph().as_default():
        cifar_dataset = data_helpers.load_data() # Loads training images/label + test images/label
        image = tf.placeholder(tf.float32,[None,3072])
        label = tf.placeholder(tf.int64,[None])


        # Construct the graph
        out,regularizer = cifar.inference(image)

        # Add op for loss
        loss = cifar.loss(out,regularizer,label)

        # Add op for optimization for each training step
        train_step = cifar.create_train_step(loss)

        # Add op for evaluating training accuracy
        accuracy = cifar.evaluate(out,label)

        # Now that all the ops are defined, run the training
        init = tf.global_variables_initializer()

        # Create a "saver" to save weights and biases
        saver = tf.train.Saver()

        # Create session
        sess = tf.Session()
        sess.run(init)

        for i in range(MAX_STEPS):
            # Generate batch
            batch = np.random.choice(cifar_dataset['images_train'].shape[0], BATCH_SIZE)
            images_batch = cifar_dataset['images_train'][batch]
            label_batch = cifar_dataset['labels_train'][batch]
            # Execute the train step
            sess.run(train_step,feed_dict = {image:images_batch,label:label_batch})
            if i%100 == 0:
                train_accuracy = evaluate_batch(sess,accuracy,images_batch,label_batch,image,label)
                print('Iteration '+str(i)+' training accuracy: '+str(train_accuracy))
            if i == MAX_STEPS-1:
                # Now that training is complete, save the checkpoint file
                file_path = os.path.join(os.getcwd(),"model.cpkt")
                saver.save(sess,file_path,global_step = i)



def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run(main=main)



