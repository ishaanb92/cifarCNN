import tensorflow as tf
import numpy as np
import cifar # Contains inference(),loss(),training()
import cifar10_input # Contains functions to read CIFAR-10 bin files
import glob
import shutil
import os
import time
from datetime import datetime

MAX_STEPS = 1000000

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
        out = cifar.inference(image,training = True)

        # Add op for loss
        loss = cifar.loss(out,label)

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

        # Used to run the train_op
        class _LoggerHook(tf.train.SessionRunHook):
          """Logs loss and runtime."""

          def begin(self):
            self._step = -1
            self._start_time = time.time()

          def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(accuracy)  # Asks for loss value.

          def after_run(self, run_context, run_values):
            if self._step % 10 == 0:
              current_time = time.time()
              duration = current_time - self._start_time
              self._start_time = current_time

              accuracy_value = run_values.results
              examples_per_sec = 10 * cifar.TRAINING_BATCH_SIZE / duration
              sec_per_batch = float(duration / 10)

              format_str = ('%s: step %d, accuracy = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              print (format_str % (datetime.now(), self._step, accuracy_value,
                                   examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir= os.path.join(os.getcwd(),'train'),
            hooks=[tf.train.StopAtStepHook(last_step=MAX_STEPS),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=False)) as mon_sess:
          while not mon_sess.should_stop():
            mon_sess.run(train_step)


        # Now that training is complete, save the checkpoint file
        file_path = os.path.join(os.getcwd(),"model.cpkt")
        saver.save(sess,file_path,global_step = global_step)

def main(_):
    cleanup()
    run_training()

if __name__ == '__main__':
    tf.app.run(main=main)



