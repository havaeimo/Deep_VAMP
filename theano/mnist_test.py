from theano.tensor.signal import downsample
from theano import tensor as T
from theano.tensor.nnet import conv
import models2
import cPickle
import gzip
import os
import sys
import time
import numpy as np
import numpy
from deep_vamp import VAMP
import theano
import theano.tensor as T
from update_rules import Momentum, DecreasingLearningRate, AdaGrad, AdaDelta, RMSProp, Adam, Adam_paper, ExpDecayLearningRate

rng = numpy.random.RandomState(23455)
learning_rate=0.01
filter_shapes=[(5,5),(5,5)]
pool_size=[(2,2),(2,2)]
pool_stride=[(2,2),(2,2)]
nkerns=[16,32]
n_epochs=100
nb_channels=3
image_size=[128,64]
update_rule = "expdecay"  
decrease_constant = 0.0001
nb_classes = 2 
init_momentum=0.9
################################################################################################
################################################################################################
################################################################################################
################################################################################################

path_testset = '/home/local/USHERBROOKE/havm2701/data/DBFrames'
batch_size = 100
train = VAMP(start=0,stop=10000,image_resize=image_size,toronto_prepro=True)
valid = VAMP(start=10000,stop=12000,image_resize=image_size,toronto_prepro=True)
train_x = train.X 
train_y = np.argmax(train.y,axis=1)
valid_x = valid.X
valid_y = np.argmax(valid.y,axis=1)
test_x = valid.X
test_y = np.argmax(valid.y,axis=1)
#train_x=np.asarray(train_x,dtype='int32')
#valid_x=np.asarray(valid_x,dtype='int32')
#test_x=np.asarray(test_x,dtype='int32')
train_set_x = theano.shared(numpy.asarray(train_x,dtype=theano.config.floatX),borrow=True)   
train_set_y = theano.shared(numpy.asarray(train_y,dtype=np.int32),borrow=True)
valid_set_x = theano.shared(numpy.asarray(valid_x,dtype=theano.config.floatX),borrow=True)
valid_set_y = theano.shared(numpy.asarray(valid_y,dtype=np.int32),borrow=True)
test_set_x = theano.shared(numpy.asarray(valid_x,dtype=theano.config.floatX),borrow=True)
test_set_y = theano.shared(numpy.asarray(valid_y,dtype=np.int32),borrow=True)   
# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size



x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
index = T.lscalar()  # index to a [mini]batch                        # [int] labels

########################################################################################################
# BUILD ACTUAL MODEL #
########################################################################################################
print '... building the model'

# Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
# (28, 28) is the size of MNIST images.
layer0_input = x.reshape((batch_size, image_size[0], image_size[1],nb_channels))
layer0_input = layer0_input.dimshuffle(0,3,1,2)

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
# maxpooling reduces this further to (24/2, 24/2) = (12, 12)
# 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
layers = [models2.LeNetConvPoolLayer(rng=rng,layerIdx=0, input=layer0_input, 
                                         filter_shape=(nkerns[0],nb_channels,filter_shapes[0][0],filter_shapes[0][1]), 
                                         image_shape=(batch_size,nb_channels,image_size[0],image_size[1]), 
                                         activation=T.tanh, 
                                         pool_size=pool_size[0], 
                                         pool_stride=pool_stride[0])]
# Construct the second convolutional pooling layer
# filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
# maxpooling reduces this further to (8/2, 8/2) = (4, 4)
# 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)



for h_id in range(1,len(nkerns)):

    nb_channels_h = layers[-1].filter_shape[0]
    featuremap_shape = models2.get_channel_shape(layers[-1])
    layers += [models2.LeNetConvPoolLayer(layerIdx=h_id,rng=rng, input=layers[-1].output, 
                                          filter_shape=(nkerns[h_id],nkerns[h_id-1],filter_shapes[h_id][0],filter_shapes[h_id][1]), 
                                          image_shape=(batch_size,nkerns[h_id-1],featuremap_shape[0],featuremap_shape[1]), 
                                          activation=T.tanh, 
                                          pool_size=pool_size[h_id], 
                                          pool_stride=pool_stride[h_id])]  



# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
# or (500, 50 * 4 * 4) = (500, 800) with the default values.
#layer2_input = layers[-1].output.flatten(2)
last_conv_fm_shape = models2.get_channel_shape(layers[-1])
# construct a fully-connected sigmoidal layer
'''layers += [models2.HiddenLayer(
    rng,
    input=layers[-1].output.flatten(2),
    n_in=nkerns[-1] * last_conv_fm_shape[0] * last_conv_fm_shape[1],
    n_out=500,
    activation=T.tanh
)]'''
n_in = nkerns[-1] * last_conv_fm_shape[0] * last_conv_fm_shape[1]
# classify the values of the fully-connected sigmoidal layer
#layers += [models2.LogisticRegression(input=layers[-1].output.flatten(2), n_in=n_in, n_out=2)]
layers += [models2.ChannelLogisticRegression(layerIdx=len(nkerns)+1,rng=rng, input=layers[-1].output,
                                      filter_shape=(nb_classes,nkerns[-1],last_conv_fm_shape[0],last_conv_fm_shape[1]), 
                                      image_shape =(batch_size,nkerns[-1],last_conv_fm_shape[0],last_conv_fm_shape[1]))]
# the cost we minimize during training is the NLL of the model
cost = layers[-1].negative_log_likelihood(y)

# create a function to compute the mistakes that are made by the model
test_model = theano.function(
    [index],
    layers[-1].errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    [index],
    layers[-1].errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

# create a list of all model parameters to be fit by gradient descent


#params = layers[0].params + layers[1].params + layers[2].params + layers[3].params
params = layers[0].params
for idx in range(1,len(layers)):   
    params += layers[idx].params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i], grads[i]) pairs.
        # Initialize update_rule
      
if update_rule == "None":
    update_rule = DecreasingLearningRate(learning_rate, decrease_constant)
if update_rule == "expdecay":    
    update_rule = ExpDecayLearningRate(learning_rate, decrease_constant)
elif update_rule == "adadelta":
    update_rule = AdaDelta(decay=decrease_constant, epsilon=learning_rate)
elif update_rule == "adagrad":
    update_rule = AdaGrad(learning_rate=learning_rate)
elif update_rule == "rmsprop":
    update_rule = RMSProp(learning_rate=learning_rate, decay=decrease_constant)
elif update_rule == "adam":
    update_rule = Adam(learning_rate=learning_rate)
elif update_rule == "adam_paper":
    update_rule = Adam_paper(learning_rate=learning_rate)
elif update_rule == "momentum":
    update_rule = Momentum(learning_rate=learning_rate,init_momentum=init_momentum)

    

updates = update_rule.get_updates(zip(params, grads))



#updates = [
#    (param_i, param_i - learning_rate * grad_i)
#    for param_i, grad_i in zip(params, grads)
#]

train_model = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

############################################################
###############
# TRAIN MODEL #
###############
print '... training the model'
# early-stopping parameters
patience = 5000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                              # found
improvement_threshold = 0.995  # a relative improvement of this much is
                              # considered significant
validation_frequency = min(n_train_batches, patience / 2)
                              # go through this many
                              # minibatche before checking the network
                              # on the validation set; in this case we
                              # check every epoch

best_validation_loss = numpy.inf
test_score = 0.
start_time = time.clock()

done_looping = False
epoch = 0
while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

        minibatch_avg_cost = train_model(minibatch_index)
        # iteration number
        iter = (epoch - 1) * n_train_batches + minibatch_index

        if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = [validate_model(i)
                                 for i in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)

            print(
                'epoch %i, minibatch %i/%i, validation error %f %%' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                )
            )

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                # test it on the test set

                test_losses = [test_model(i)
                               for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)

                print(
                    (
                        '     epoch %i, minibatch %i/%i, test error of'
                        ' best model %f %%'
                    ) %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_score * 100.
                    )
                )

        if patience <= iter:
            done_looping = True
            break

end_time = time.clock()
print(
    (
        'Optimization complete with best validation score of %f %%,'
        'with test performance %f %%'
    )
    % (best_validation_loss * 100., test_score * 100.)
)
print 'The code run for %d epochs, with %f epochs/sec' % (
    epoch, 1. * epoch / (end_time - start_time))
print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.1fs' % ((end_time - start_time)))
