import tensorflow as tf
import numpy as np
import data_helpers
import cifar
import os
import sys

TEST_BATCH_SIZE = 10000

def run_test():
    with tf.Graph().as_default():

        cifar10_data = data_helpers.load_data()
        test_images = cifar10_data['images_test']
        test_labels = cifar10_data['labels_test']

        # Create placeholders
        images = tf.placeholder(tf.float32,[TEST_BATCH_SIZE,3072])
        labels = tf.placeholder(tf.int64,[TEST_BATCH_SIZE])
        # Create op to compute output of network
        out,regularizer,keep_prob_pool2,keep_prob_fc1 = cifar.inference(images,training = False)
        result = cifar.evaluate(out,labels)

        # Load saved data
        loader = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(os.getcwd())

        if ckpt and ckpt.model_checkpoint_path:
            loader.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Could not load saved data \n"
            sys.exit()
        test_accuracy = sess.run(result,feed_dict = {images:test_images,labels:test_labels,keep_prob_pool2 : 1.0,keep_prob_fc1 : 1.0,keep_prob_fc2 : 1.0})
        print 'Test accuracy for CIFAR-10 is %.04f'%(test_accuracy)

def main(_):
    run_test()

if __name__ == '__main__':
    tf.app.run(main = main)



