
import numpy as np
#from PIL import Image
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#from util import tile_raster_images
tf.disable_v2_behavior()


from RBM import RBM # Class for RBM implementation
from NN import NN   # Class for Feed forward Network implementation

    
from tensorflow.examples.tutorials.mnist import input_data            #Getting the MNIST data provided by Tensorflow

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)        #Loading in the mnist data
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


RBM_hidden_sizes = [500, 200 , 50 ] #create 3 layers of RBM with size 500, 200 and 50

#Since we are training, set input as training data
inpX = trX

#Create list to hold our RBMs
rbm_list = []

#Size of inputs is the number of inputs in the training set
input_size = inpX.shape[1]

#For each RBM we want to generate
for i, size in enumerate(RBM_hidden_sizes):
    print ('RBM: ',i,' ',input_size,'->', size)
    rbm_list.append(RBM(input_size, size))
    input_size = size


    #For each RBM in our list
for rbm in rbm_list:
    print ('New RBM:')
    #Train a new one
    rbm.train(inpX) 
    #Return the output layer
    inpX = rbm.rbm_outpt(inpX)  #THE INPUT OF NEXT RBM IS GONIG TO BE THE OUTPUT OF THE PREVIOUS ONE


nNet = NN(RBM_hidden_sizes, trX, trY)
nNet.load_from_rbms(RBM_hidden_sizes,rbm_list)
nNet.train()