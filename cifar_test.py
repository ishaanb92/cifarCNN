import tensorflow as tf
import numpy as np
import cifar
import os
import sys
import math

runOnce = True

def doEval(loader,pred_accu_op,summary_op,summary_writer):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.join(os.getcwd(),'train')) # Load checkpoint file
        # Load saved training vars
        if ckpt and ckpt.model_checkpoint_path:
            loader.restore(sess,ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('Checkpoint file not found \n')
            sys.exit()

        coord = tf.train.Coordinator()
        try:
          threads = []
          for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))

            num_iter = int(cifar.NUM_TEST_EXAMPLES/cifar.TRAINING_BATCH_SIZE)
            true_count = 0
            total_samples = num_iter*cifar.TRAINING_BATCH_SIZE
            step = 0
            print ('Starting iterations \n')
            while step < num_iter:
                print('On iteration %d \n'%step)
                preds = sess.run([pred_accu_op])
                true_count  += np.sum(preds)
                step += 1
                print('True count for iteration %d = %d'%((step-1),true_count))
            print('Ran batches \n')
            predication = true_count/total_samples
            print('Prediction accuracy of the model = %.3f'%(prediction))
            # Generate summary files
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
          coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def runTest():

    with tf.Graph().as_default() as g:
        # Get the test images and labels
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

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(os.getcwd(),'test'),g)
        # Evaluate the test-set in batches
        while True:
            doEval(loader,pred_accu_op,summary_op,summary_writer)
            if runOnce:
                break

def main(argv=None):
    runTest()

if __name__ == '__main__':
    tf.app.run(main = main)



