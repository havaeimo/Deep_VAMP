import pdb
import numpy
import models
from models import myReLu
from deep_vamp import VAMP
import argparse

import os
import theano.tensor as T
import numpy as np
import theano
import time
import utilities
from mnist_model import DeepVamp
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
    dataset = '/home/local/USHERBROOKE/havm2701/git.repos/Deep_VAMP/theano/mnist.pkl.gz'
    datasets = models.load_data(dataset)
    pdb.set_trace()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    pdb.set_trace()

    #train = VAMP(start=0,stop=10000,image_resize=[128,64],toronto_prepro=True)
    #valid = VAMP(start=10000,stop=12000,image_resize=[128,64],toronto_prepro=True)
    #valid = valid.get_reshaped_images()
    #train = train.get_reshaped_images()
    dataset ={}
    #train.y=np.argmax(train.y,axis=1)
    #train.y = np.asarray(train.y,dtype=np.32)
    #valid.y=np.argmax(valid.y,axis=1)
    #valid.y = np.asarray(valid.y,dtype=np.32)
    pdb.set_trace()
    dataset['train'] = datasets[0]
    dataset['valid'] = (valid_set_x, valid_set_y)
    dataset['nb_classes'] = 10
    classifier = DeepVamp(dataset,random_seed=1234,
                          learning_rate=0.01,
                          decrease_constant=0.95,
                          hidden_size =[64],
                          filter_shapes=[(5,5)],
                          pool_size=[(4,4)],
                          pool_stride=[(4,4)],
                          update_rule = "None",
                          activation=[T.tanh],
                          batch_size=100
                          )

    # pdb.set_trace()
    #classifier.train(0).shape
    #pdb.set_trace()
    #test_dataset = train.X[:100,...]
    #outt = classifier.use(test_dataset.transpose(0,3,1,2))

    classifier = utilities.train(classifier,max_epochs=500)
    #import pdb            
    #pdb.set_trace()
    #outt = classifier.use(test_dataset.transpose(0,3,1,2))
    #import cPickle
    #f = file('obj.save', 'wb')
    #cPickle.dump(classifier, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #f.close()




