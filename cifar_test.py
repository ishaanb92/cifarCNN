import tensorflow as tf
import numpy as np
import cifar
import os
import sys

TEST_BATCH_SIZE = 10000

def run_test():

    with tf.Graph().as_default():

        image, labels = cifar.inputs(eval_data=True)
        # Construct the inference graph
        out = cifar.inference(image)
        # Define prediction accuracy op
        pred_accu_op = tf.nn.in_top_k(out,labels,1)

        # Create pointer to saved moving averages of trained vars
        variable_averages = tf.train.ExponentialMovingAverage(cifar.MOVING_AVERAGES_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

        # Create a loader
        loader = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.join(os.getcwd(),'train')) # Load checkpoint file
            # Load saved training vars
            if ckpt and ckpt.model_checkpoint_path:
                loader.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('Checkpoint file not found \n')
                sys.exit()
            num_iterations = int(TEST_BATCH_SIZE/cifar.TRAINING_BATCH_SIZE)
            true_count = 0
            total_samples = num_iterations*cifar.TRAINING_BATCH_SIZE
            step = 0

            print ('Starting iterations \n')
            while step < num_iterations:
                print('On iteration %d'%step)
                preds = sess.run([pred_accu_op])
                true_count  += np.sum(preds)
                step += 1
                print('True count for iteration %d = %d'%((step-1),true_count))

            print('Ran batches \n')
            print('Prediction accuracy of the model = %.3f'%(preds))

def main(_):
    run_test()

if __name__ == '__main__':
    tf.app.run(main = main)



