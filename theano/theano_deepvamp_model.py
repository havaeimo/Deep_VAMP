import numpy
import models
from deep_vamp import VAMP
import argparse
import pdb
import os
import theano.tensor as T
import numpy as np
import theano

class DeepVamp(object):

    def __init__(self, input, random_seed,learning_rate=0.01, hiddensize=[500,100]):
    	  # the data is presented as rasterized images
    	  # the labels are presented as 1D vector of

    	#input_size = output_size = dataset['input_size']
    	self.rng = np.random.mtrand.RandomState(random_seed)
    	self.learning_rate = learning_rate
    	self.layers = [models.LeNetConvPoolLayer(self.rng, input, filter_shape=(3,3,5,5), image_shape=(100,3,32,32), activation=T.tanh, pool_size=(4, 4), pool_stride=(1,1))]

    	self.layers += [models.OutputLayer(input=self.layers[-1].out, rng=self.rng,filter_shape=(2,3,21,21), image_shape=(100,3,21,21))]

    	cost_obj = models.Cost(self.layers[-1].out, target)
    	self.cost = cost_obj.out

        layer_parameters = []
        for i in range(2):
            layer_parameters += self.layers[i].params

        parameters_gradiants = T.grad(self.cost, layer_parameters)    
        updates = []
        for param, param_gradiant in zip(layer_parameters, parameters_gradiants):
            updates += (param, param - self.learning_rate * param_gradiant)


        #self.train = theano.function([input],cost, updates=updates)    

'''
        self.train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    
        index = T.lscalar()
        self.learn = theano.function(name='learn',
                                     inputs=[index],
                                     outputs=loss,
                                     updates=updates,
                                     givens={input: dataset['train']['data'][index * batch_size:(index + 1) * batch_size], target: dataset['train']['target'][index * batch_size:(index + 1) * batch_size]}) 
    


        self.use = theano.function(name='use',
                                   inputs=[input],
                                   outputs=cost) 
    '''

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
    x = T.tensor4('x')
    target = T.matrix('target')
    index = T.lscalar() 
    path_testset = '/home/local/USHERBROOKE/havm2701/data/DBFrames'
    batch_size = 100
    train = VAMP()
    test_x = train.X[:100,:].reshape(100,3,32,32)

    test_set_y = train.y[:100,:]

    classifier = DeepVamp(input=x,random_seed=1234)
    test_set_x = theano.shared(numpy.asarray(test_x,dtype=theano.config.floatX),borrow=True)   
    test_model = theano.function(
    inputs=[index],
    outputs=classifier.layers[-2].out,
    givens={
        x: test_set_x[index * batch_size:(index + 1) * batch_size],
        #y: test_set_y[index * batch_size:(index + 1) * batch_size]
    }
    )
    # pdb.set_trace()
    print test_model(0)
