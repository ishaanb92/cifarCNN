import tensorflow as tf
import numpy as np
import cifar # Contains inference(),loss(),training()
import cifar10_input # Contains functions to read CIFAR-10 bin files
import glob
import shutil
import os

MAX_STEPS = 250*1000

# clean up dir
def cleanup():
    files = glob.glob("model"+"*")
    for f in files:
        os.remove(f)
    if os.path.isdir(os.path.join(os.getcwd(),'train')):
        shutil.rmtree('./train')
    if os.path.isdir(os.path.join(os.getcwd(),'test')):
        shutil.rmtree('./test')

# Create computation graph for training
def run_training():

    with tf.Graph().as_default():


        # Generate a batch
        image, label = cifar.distorted_inputs()

        global_step = tf.contrib.framework.get_or_create_global_step()

        # Construct the graph
        out,regularizer = cifar.inference(image,training = True)

        # Add op for loss
        loss = cifar.loss(out,regularizer,label)

        # Add op for optimization for each training step
        train_step,variables_to_restore = cifar.create_train_step(loss,global_step)

        # Add op for evaluating training accuracy
        accuracy = cifar.evaluate(out,label)

        # Now that all the ops are defined, run the training
        init = tf.global_variables_initializer()

        # Create a "saver" to save running avg of weights and biases
        saver = tf.train.Saver(variables_to_restore)

        # Create session
        sess = tf.Session()
        sess.run(init)

        # Handling visualizations
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.getcwd() + '/train',sess.graph)
        test_writer = tf.summary.FileWriter(os.getcwd() + '/test',sess.graph)

        steps_per_epoch_train = cifar.NUM_TRAINING_EXAMPLES/cifar.TRAINING_BATCH_SIZE # Train ops run per training epoch
        steps_per_epoch_test = cifar.NUM_TEST_EXAMPLES/cifar.TRAINING_BATCH_SIZE # Number of evaluations per epoch for test examples
        num_epochs = MAX_STEPS/steps_per_epoch_train # Number of training epochs

        for i in range(num_epochs):
            # Generate batch
            for j in range(steps_per_epoch_train):
                # Execute the train step
                summary,_ = sess.run([merged,train_step]) # The model "experiences" a new batch every step in an epoch
            # Record current training loss every epoch
            train_accuracy = sess.run([accuracy])
            train_writer.add_summary(summary,i)
            print('Epoch '+str(i)+' training accuracy: '+str(train_accuracy))

        # Now that training is complete, save the checkpoint file
        file_path = os.path.join(os.getcwd(),"model.cpkt")
        saver.save(sess,file_path,global_step = i)



def main(_):
    cleanup()
    run_training()

if __name__ == '__main__':
    tf.app.run(main=main)



