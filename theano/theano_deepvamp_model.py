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

class DeepVamp(object):
    def __init__(self, dataset, random_seed,learning_rate=0.001, hidden_size=[3],filter_shapes=[(5,5)],pool_size=[(4,4)],activation=[T.tanh],batch_size=100):
    	  # the data is presented as rasterized images
    	  # the labels are presented as 1D vector of
        
    	#input_size = output_size = dataset['input_size']
        assert len(hidden_size) == len(filter_shapes)
    
        input = T.tensor4('input')
        target = T.matrix('target')
        
        train = dataset['train']
        valid = dataset['valid']
        train_x = train.X
        train_x = train_x.transpose(0,3,1,2)
        valid_x = valid.X
        valid_x = valid_x.transpose(0,3,1,2)
        valid_y = valid.y
        train_y = train.y     
        image_size = train_x.shape[2:]
        nb_channels = train_x.shape[1]

    	self.rng = np.random.mtrand.RandomState(random_seed)
    	self.learning_rate = learning_rate
        
        #Build Model
      
    	self.layers = [models.LeNetConvPoolLayer(self.rng, input, filter_shape=(hidden_size[0],nb_channels,filter_shapes[0][0],filter_shapes[0][1]), image_shape=(batch_size,nb_channels,image_size[0],image_size[1]), activation=activation[0], pool_size=pool_size[0], pool_stride=(1,1))]
        for h_id in range(len(hidden_size)-1):
                 nb_channels_h = self.layers[-1].nb_filters
                 featuremap_shape = self.layers[-1].out.shape[2,:]
                 self.layers = [models.LeNetConvPoolLayer(self.rng, input, filter_shape=(hidden_size[h_id],nb_channels_h,filter_shapes[h_id][0],filter_shapes[h_id][1]), image_shape=(batch_size,nb_channels_h,featuremap_shape[0],featuremap_shape[1]), activation=activation[h_id], pool_size=pool_size[h_id], pool_stride=(1,1))]



 
       self.layers += [models.OutputLayer(input=self.layers[-1].out, rng=self.rng,filter_shape=(2,3,7,7), image_shape=(100,3,7,7))]
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



    index = T.lscalar() 
    path_testset = '/home/local/USHERBROOKE/havm2701/data/DBFrames'
    batch_size = 100
    train = VAMP(start=0,stop=10000)
    valid = VAMP(start=10000,stop=12000)
    valid = valid.get_reshaped_images()
    train = train.get_reshaped_images()
    dataset ={}
    dataset['train'] = train
    dataset['valid'] = valid
    classifier = DeepVamp(dataset,random_seed=1234)

    # pdb.set_trace()
    classifier.train(0).shape
    #pdb.set_trace()





   ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    #n_train_batches = 20                               # considered significant
    validation_frequency = min(classifier.n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    n_epochs = 100
    epoch = 0
    done_looping = False
   
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(classifier.n_train_batches):

            minibatch_avg_cost = classifier.train(minibatch_index)
            # iteration number
            iter = (epoch - 1) * classifier.n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [classifier.valid(i) for i
                                     in xrange(classifier.n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%, training error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        classifier.n_train_batches,
                        this_validation_loss * 100.,
                        minibatch_avg_cost 
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
             #       test_losses = [test_model(i) for i
             #                      in xrange(n_test_batches)]
             #       test_score = numpy.mean(test_losses)
             # 
             #       print(('     epoch %i, minibatch %i/%i, test error of '
             #              'best model %f %%') %
             #             (epoch, minibatch_index + 1, n_train_batches,
             #              test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
