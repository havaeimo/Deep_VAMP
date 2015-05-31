import numpy
import models
from models import relu
from deep_vamp import VAMP
import argparse

import os
import theano.tensor as T
import numpy as np
import theano
import time
import utilities
from Deep_vamp_model import DeepVamp
from scipy.misc import imsave
if __name__ == "__main__":

    #parser = argparse.ArgumentParser(description='Generate the DICE score for a BrainSet')
    #parser.add_argument('model', type=argparse.FileType('r'),
     #                   help='A serialized pylearn2 model.')
    #####parser.add_argument('testset', type=str,
    #                    help='The path to test images.'),
    #parser.add_argument('patch_shape', type=int,
    #                    help='The size of the input patch window.'),
    #parser.add_argument('label_patch_shape', type=int,
    #                    help='The size of the predicted patch window.'),
    #parser.add_argument('num_channels', type=int,
    #                    help='Number of channels in the dataset.'),
    #args = parser.parse_args()
    #path_testset = self.testset



    
    path_testset = '/home/local/USHERBROOKE/havm2701/data/DBFrames'
    batch_size = 100
    train = VAMP(start=0,stop=10000,image_resize=[128,64],toronto_prepro=True)
    valid = VAMP(start=10000,stop=12000,image_resize=[128,64],toronto_prepro=True)
    valid = valid.get_reshaped_images()
    train = train.get_reshaped_images()
    dataset ={}
    dataset['train'] = train
    dataset['valid'] = valid
    dataset['nb_classes'] = train.nb_classes
    classifier = DeepVamp(dataset,random_seed=1234,
                          learning_rate=0.01,
                          hidden_size =[64,32],
                          filter_shapes=[(4,4),(2,2)],
                          pool_size=[(4,4),(2,2)],
                          pool_stride=[(2,2),(2,2)],
                          activation=[T.tanh,T.tanh],
                          batch_size=100
                          )

    # pdb.set_trace()
    classifier.train(0).shape
    #pdb.set_trace()


    utilities.train(classifier,max_epochs=200)







