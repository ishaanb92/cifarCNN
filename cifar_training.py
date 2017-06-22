import tensorflow as tf
import numpy as np
import cifar # Contains inference(),loss(),training()
import cifar10_input # Contains functions to read CIFAR-10 bin files
import glob
import shutil
import os
import time
from datetime import datetime
import math

MAX_STEPS = 1000*1000

# clean up dir
def cleanup():
    files = glob.glob("model"+"*")
    for f in files:
        os.remove(f)
    if os.path.isdir(os.path.join(os.getcwd(),'train')):
        shutil.rmtree('./train')
    if os.path.isdir(os.path.join(os.getcwd(),'test')):
        shutil.rmtree('./test')

def evaluate_batch(sess,accuracy_op,num_examples):
    # Inputs Args:
        # sess : Handle to the current session
        # accuracy_op : Operation in the computational graph to get the accuracy
        # num_examples : Number of examples
    # Returns :
        # fraction of correctly classified images

    num_iter = int(math.ceil(num_examples/cifar.TRAINING_BATCH_SIZE))
    true_count = 0
    total_sample_count = num_iter*cifar.TRAINING_BATCH_SIZE
    for step in range(num_iter):
        predictions = sess.run([accuracy_op])
        # predictions is a bool array of length "BATCH_SIZE"
        true_count += np.sum(predictions)

    return true_count/total_sample_count

# Create computation graph for training
def run_training():

    with tf.Graph().as_default():
        with tf.variable_scope("model") as scope:
            #Generate a batch
            global_step = tf.Variable(0, trainable=False)

            image_train, label_train = cifar.distorted_inputs()

            image_test, label_test = cifar.inputs(eval_data=True)

            # Construct the graph
            out_train = cifar.inference(image_train)

            scope.reuse_variables()

            # Inference graph for test images
            out_test = cifar.inference(image_test)

            # Add op for loss
            loss = cifar.loss(out_train,label_train)

            # Add op for evaluating training accuracy
            accuracy = cifar.evaluate(out_train,label_train)

            # Add op for evaluating accuracy on test data
            accuracy_test = cifar.evaluate(out_test,label_test)

            # Placeholders for train/test precision
            summary_train_prec = tf.placeholder(tf.float32)
            summary_eval_prec  = tf.placeholder(tf.float32)
            tf.summary.scalar('accuracy/train', summary_train_prec)
            tf.summary.scalar('accuracy/eval', summary_eval_prec)

            # Add op for optimization for each training step
            train_step = cifar.create_train_step(loss,global_step)

            # Create a saver.
            saver = tf.train.Saver(tf.global_variables())
            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            # Create session
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            sess.run(init)

            # Starts the threads
            tf.train.start_queue_runners(sess=sess)

            summary_writer = tf.summary.FileWriter(os.path.join(os.getcwd(),'train'),graph=sess.graph)

            for step in range(MAX_STEPS):
                _,loss_value = sess.run([train_step,loss])
                if step%10 == 0:
                    print('TRAINING :: loss = %.2f step = %d \n'%(loss_value,step))
                    # Evaluate accuracy
                    prec_train = evaluate_batch(sess,accuracy,cifar.NUM_TRAINING_EXAMPLES)
                    prec_eval  = evaluate_batch(sess,accuracy_test,cifar.NUM_TEST_EXAMPLES)
                    print('accuracy train = %.3f' % (prec_train))
                    print('accuracy eval  = %.3f' % (prec_eval))
                if step % 100 == 0:
                    summary_str = sess.run(summary_op, feed_dict={summary_train_prec: prec_train,summary_eval_prec:  prec_eval})
                    summary_writer.add_summary(summary_str, step)
                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == MAX_STEPS:
                    checkpoint_path = os.path.join(os.getcwd(),'train','model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

def main(_):
    cleanup()
    run_training()

if __name__ == '__main__':
    tf.app.run(main=main)



