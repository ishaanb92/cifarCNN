The goal here is to see whether augmenting the training data using generative models can improve classifcation accuracy for the test images in the absence of deliberate noise addition. 

The code to pre-process the CIFAR-10 data-set (unpickle and load) has been forked from https://github.com/wolfib/image-classification-CIFAR10-tf

The base architecture for the classifier is based on the CIFAR-10 classifier present in the TensorFlow tutorials, just more amenable to change. For the original code (much more polished), please refer : https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10. In the original TF code, training images are distorted using random crops and other techniques. Input images in this code are not manipulated before training.

TODOS:
1. Make code modular, clear distinction between training and evaluation
2. Use a adversarial inputs to augment training data 
