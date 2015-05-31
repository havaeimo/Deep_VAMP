import numpy
import models
from models import relu
from deep_vamp import VAMP
import argparse
import pdb
import os
import theano.tensor as T
import numpy as np
import theano
import time
import utilities
class DeepVamp(object):
    def __init__(self, dataset, random_seed,learning_rate, hidden_size,filter_shapes,pool_size,pool_stride,activation,batch_size=100):
    	  # the data is presented as rasterized images
    	  # the labels are presented as 1D vector of
        
    	#input_size = output_size = dataset['input_size']
        assert len(hidden_size) == len(filter_shapes)
        index = T.lscalar() 
        input = T.tensor4('input')
        target = T.matrix('target')
        nb_classes = dataset['nb_classes']
        train = dataset['train']
        valid = dataset['valid']
        train_x = train.X
        train_x = train_x.transpose(0,3,1,2)
        from scipy.misc import imsave
        ex = train.X[0,0,:,:]
        imsave('test_new.png', ex)
        pdb.set_trace()
        valid_x = valid.X
        valid_x = valid_x.transpose(0,3,1,2)
        valid_y = valid.y
        train_y = train.y     
        image_size = train_x.shape[2:]
        nb_channels = train_x.shape[1]
        
    	self.rng = np.random.mtrand.RandomState(random_seed)
    	self.learning_rate = learning_rate
        
        #Build Model
        def get_input_shape(layer):
            filter_shape = layer.filter_shape[2:]
            image_shape = layer.image_shape[2:]
            pool_size = layer.pool_size
            pool_stride = layer.pool_stride
            conv_stride = layer.conv_stride
            def reduction(image,kernel,stride):
                return (image - kernel)/stride +1
            post_conv = (reduction(image_shape[0],filter_shape[0],conv_stride[0]) ,reduction(image_shape[1],filter_shape[1],conv_stride[1]))
            post_pool = (reduction(post_conv[0],pool_size[0],pool_stride[0]) ,reduction(post_conv[1],pool_size[1],pool_stride[1])) 
            
            return post_pool      


    	self.layers = [models.LeNetConvPoolLayer(self.rng, input, filter_shape=(hidden_size[0],nb_channels,filter_shapes[0][0],filter_shapes[0][1]), image_shape=(batch_size,nb_channels,image_size[0],image_size[1]), activation=activation[0], pool_size=pool_size[0], pool_stride=pool_stride[0])]
        for h_id in range(1,len(hidden_size)):
                 nb_channels_h = self.layers[-1].filter_shape[0]
                 featuremap_shape = get_input_shape(self.layers[-1])
                 self.layers = [models.LeNetConvPoolLayer(self.rng, self.layers[-1].out, filter_shape=(hidden_size[h_id],hidden_size[h_id-1],filter_shapes[h_id][0],filter_shapes[h_id][1]), image_shape=(batch_size,hidden_size[h_id-1],featuremap_shape[0],featuremap_shape[1]), activation=activation[h_id], pool_size=pool_size[h_id], pool_stride=pool_stride[h_id])]

        output_filter_shape = get_input_shape(self.layers[-1])
        nb_channels_out = self.layers[-1].filter_shape[0]
        final_h_shape = output_filter_shape
        self.layers += [models.OutputLayer(input=self.layers[-1].out, rng=self.rng,filter_shape=(nb_classes,nb_channels_out,output_filter_shape[0],output_filter_shape[1]), image_shape=(batch_size,nb_channels_out,final_h_shape[0],final_h_shape[1]))]
     	cost_obj = models.Cost(self.layers[-1].out, target)
    	self.cost = cost_obj.out

        layer_parameters = []
        for i in range(2):
            layer_parameters += self.layers[i].params

        parameters_gradiants = T.grad(self.cost, layer_parameters)    
        updates = []
        for param, param_gradiant in zip(layer_parameters, parameters_gradiants):
            updates += [(param, param - self.learning_rate * param_gradiant)]


        #self.train = theano.function([input],cost, updates=updates)
        train_set_x = theano.shared(numpy.asarray(train_x,dtype=theano.config.floatX),borrow=True)   
        train_set_y = theano.shared(numpy.asarray(train_y,dtype=theano.config.floatX),borrow=True)
        valid_set_x = theano.shared(numpy.asarray(valid_x,dtype=theano.config.floatX),borrow=True)
        valid_set_y = theano.shared(numpy.asarray(valid_y,dtype=theano.config.floatX),borrow=True)     
        #train_set_x = train_set_x.dimshuffle(0,3,1,2)
        #valid_set_x = valid_set_x.dimshuffle(0,3,1,2)

        self.n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        
        self.train = theano.function(
        inputs=[index],
        outputs=self.cost,
        updates=updates,
        givens={
            input: train_set_x[index * batch_size:(index + 1) * batch_size],
            target: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
        )

        self.valid = theano.function(
        inputs=[index],
        outputs=self.cost,
        givens={
            input: valid_set_x[index * batch_size:(index + 1) * batch_size],
            target: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
        )