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
from pylearn2.datasets.deep_vamp import VAMP
import theano
import theano.tensor as T
from update_rules import Momentum, DecreasingLearningRate, AdaGrad, AdaDelta, RMSProp, Adam, Adam_paper, ExpDecayLearningRate
from utils import Timer
rng = numpy.random.RandomState(23455)
learning_rate=0.001
filter_shapes=[(5,5),(5,5)]
pool_size=[(2,2),(2,2)]
pool_stride=[(2,2),(2,2)]
nkerns=[32]
n_epochs=100
nb_channels=3
image_size=[128,64]
update_rule = "expdecay"  
decrease_constant = 0.0001
nb_classes = 2 
init_momentum = 0.9
gamma = 100
################################################################################################
################################################################################################
################################################################################################
################################################################################################

#path_testset = '/home/local/USHERBROOKE/havm2701/data/Data/DBFrames'
batch_size = 100

# Load the source dataset
with Timer("loading datasets"):
  train_s = VAMP(start=0,stop=1000,image_resize=image_size,toronto_prepro=True,read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/PN_VIRTUAL')
  train_s_x = train_s.X 
  train_s_yf = np.argmax(train_s.y,axis=1)
  train_s_yd = np.ones((len(train_s_yf)), dtype=np.int32)

  # LOad the target dataset
  train_t = VAMP(start=0,stop=1000,image_resize=image_size,toronto_prepro=True,read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/PN_REAL')
  train_t_x = train_t.X 
  train_t_yd = np.zeros((len(train_s_yf)), dtype=np.int32)

  valid = VAMP(start=0,stop=1000,image_resize=image_size,toronto_prepro=True,read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/TRAIN_REAL_PN_VIRTUAL')
  valid_x = valid.X
  valid_y = np.argmax(valid.y,axis=1)
  test_x = valid.X
  test_y = np.argmax(valid.y,axis=1)

 
with Timer("defining symbolic variables for dataset"):

  train_s_set_x = theano.shared(numpy.asarray(train_s_x,dtype=theano.config.floatX),borrow=True)   
  train_s_set_yf = theano.shared(numpy.asarray(train_s_yf,dtype=np.int32),borrow=True)
  train_s_set_yd = theano.shared(train_s_yd,borrow=True)

  train_t_set_x = theano.shared(numpy.asarray(train_t_x,dtype=theano.config.floatX),borrow=True)   
  train_t_set_yd = theano.shared(train_t_yd,borrow=True)


  valid_set_x = theano.shared(numpy.asarray(valid_x,dtype=theano.config.floatX),borrow=True)
  valid_set_y = theano.shared(numpy.asarray(valid_y,dtype=np.int32),borrow=True)
  test_set_x = theano.shared(numpy.asarray(valid_x,dtype=theano.config.floatX),borrow=True)
  test_set_y = theano.shared(numpy.asarray(valid_y,dtype=np.int32),borrow=True)   
  # compute number of minibatches for training, validation and testing
  n_train_batches = train_s_set_x.get_value(borrow=True).shape[0] / batch_size
  n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
  n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size



  x_s = T.matrix('x_s')   # the data is presented as rasterized images
  yf_s = T.ivector('yf_s')  # the labels are presented as 1D vector of
  yd_s = T.ivector('yd_s')  # the labels are presented as 1D vector of
  x_t = T.matrix('x_t')   
  yd_t = T.ivector('yd_t')

  index = T.lscalar()  # index to a [mini]batch                        # [int] labels
  p = T.lscalar() 
########################################################################################################
# BUILD ACTUAL MODEL #
########################################################################################################
# Forward prop for the Lf_s
feature_representation_layers = [models2.LeNetConvPoolLayer(rng=rng,layerIdx=0,
                                         filter_shape=(nkerns[0],nb_channels,filter_shapes[0][0],filter_shapes[0][1]), 
                                         image_shape=(batch_size,nb_channels,image_size[0],image_size[1]), 
                                         activation=T.tanh, 
                                         pool_size=pool_size[0], 
                                         pool_stride=pool_stride[0])]
for h_id in range(1,len(nkerns)):
    nb_channels_h = feature_representation_layers[-1].filter_shape[0]
    featuremap_shape = models2.get_channel_shape(feature_representation_layers[-1])
    feature_representation_layers += [models2.LeNetConvPoolLayer(layerIdx=h_id,rng=rng, 
                                          filter_shape=(nkerns[h_id],nkerns[h_id-1],filter_shapes[h_id][0],filter_shapes[h_id][1]), 
                                          image_shape=(batch_size,nkerns[h_id-1],featuremap_shape[0],featuremap_shape[1]), 
                                          activation=T.tanh, 
                                          pool_size=pool_size[h_id], 
                                          pool_stride=pool_stride[h_id])]  

last_conv_fm_shape = models2.get_channel_shape(feature_representation_layers[-1])
n_in = nkerns[-1] * last_conv_fm_shape[0] * last_conv_fm_shape[1]

classification_branch = feature_representation_layers + [models2.ChannelLogisticRegression(layerIdx=len(nkerns)+1,rng=rng, 
                                      filter_shape=(nb_classes,nkerns[-1],last_conv_fm_shape[0],last_conv_fm_shape[1]), 
                                      image_shape =(batch_size,nkerns[-1],last_conv_fm_shape[0],last_conv_fm_shape[1]))]


x_s_input = x_s.reshape((batch_size, image_size[0], image_size[1],nb_channels))
x_s_input = x_s_input.dimshuffle(0,3,1,2)

next_layer_input = x_s_input
for layer in classification_branch:
  next_layer_input = layer.fprop(next_layer_input)

p_y_given_x = next_layer_input.copy()

Lf_s = classification_branch[-1].negative_log_likelihood(next_layer_input,yf_s)

#Forward prop for Ld_s

domainadapt_branch = feature_representation_layers + [models2.ChannelLogisticRegression(layerIdx=len(nkerns)+2,rng=rng,
                                      filter_shape=(nb_classes,nkerns[-1],last_conv_fm_shape[0],last_conv_fm_shape[1]), 
                                      image_shape =(batch_size,nkerns[-1],last_conv_fm_shape[0],last_conv_fm_shape[1]))]

x_s_input = x_s.reshape((batch_size, image_size[0], image_size[1],nb_channels))
x_s_input = x_s_input.dimshuffle(0,3,1,2)
domainadapt_branch[0].input = x_s_input
next_layer_input = x_s_input
for layer in domainadapt_branch:
  next_layer_input = layer.fprop(next_layer_input)


Ld_s = domainadapt_branch[-1].negative_log_likelihood(next_layer_input,yd_s)
  
# Forward prop for Ld_T
x_t_input = x_t.reshape((batch_size, image_size[0], image_size[1],nb_channels))
x_t_input = x_t_input.dimshuffle(0,3,1,2)

next_layer_input = x_t_input
for layer in domainadapt_branch:
  next_layer_input = layer.fprop(next_layer_input)

Ld_t = domainadapt_branch[-1].negative_log_likelihood(next_layer_input, yd_t)

#import theano.printing as printing
#Ld_s = printing.Print('text')(Ld_s)

#Ld_s = printing.Print('text')(Ld_s)
#Lf_s = printing.Print('text')(Lf_s)

#Different cost functions |Lf_s: the cost fucntion associated to the regulare fprop of source domain example
#                         |Ld_s: the source domain sensitive loss function
#                         |Ld_t: the target domain sensitive loss function
# since the domain losses are maximizing we use negative loss in the cost formula
lambda_p = 2/(1+T.exp(-gamma * p)) - 1
ccost = Lf_s - lambda_p *( Ld_s + Ld_t)
#ccost = -(lambda_p+1) *( Ld_s + Ld_t)
#ccost = printing.Print('text')(ccost)
# create a list of all model parameters to be fit by gradient descent
import theano.printing as printing
ccost = printing.Print('text')(ccost)

params_w = feature_representation_layers[0].params
for idx in range(1,len(feature_representation_layers)):
    params_w += feature_representation_layers[idx].params



params_f = classification_branch[-1].params
params_d = domainadapt_branch[-1].params
# create a list of gradients for all model parameters

params = params_w + params_f + params_d
#params = params_w + params_d
grads = T.grad(ccost, params)

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
with Timer("compiling train method"):
  train_model = theano.function(
      inputs=[index,p],
      outputs=ccost,
      updates=updates,
      givens={

              yd_t: train_t_set_yd[index * batch_size: (index + 1) * batch_size],
              x_s:  train_s_set_x[index * batch_size: (index + 1) * batch_size],
              yf_s: train_s_set_yf[index * batch_size: (index + 1) * batch_size],
              yd_s: train_s_set_yd[index * batch_size: (index + 1) * batch_size],
              x_t:  train_t_set_x[index * batch_size: (index + 1) * batch_size],
              
              
      },)
    #on_unused_input='ignore'
'''
# create a function to compute the mistakes that are made by the model
with Timer("compiling test method"):
  test_model = theano.function(
      inputs=[index,p],
      outputs=ccost,#classification_branch[-1].errors(p_y_given_x,yf_s),
      givens={
              x_s  :train_s_set_x[index * batch_size: (index + 1) * batch_size],
              yf_s :train_s_set_yf[index * batch_size: (index + 1) * batch_size],
              yd_s :train_s_set_yd[index * batch_size: (index + 1) * batch_size],
              x_t  :train_t_set_x[index * batch_size: (index + 1) * batch_size],
              yd_t :train_t_set_yd[index * batch_size: (index + 1) * batch_size]
      })

with Timer("compiling valid method"):
  validate_model = theano.function(
      [index,p],
      ccost,#classification_branch[-1].errors(p_y_given_x,yf_s),
      givens={
              x_s  :train_s_set_x[index * batch_size: (index + 1) * batch_size],
              yf_s :train_s_set_yf[index * batch_size: (index + 1) * batch_size],
              yd_s :train_s_set_yd[index * batch_size: (index + 1) * batch_size],
              x_t  :train_t_set_x[index * batch_size: (index + 1) * batch_size],
              yd_t :train_t_set_yd[index * batch_size: (index + 1) * batch_size]
      })'''
'''
cost_f_s = theano.function(
    [index],
    Lf_s,#classification_branch[-1].errors(p_y_given_x,yf_s),
    givens={
            x_s  :train_s_set_x[index * batch_size: (index + 1) * batch_size],
            yf_s :train_s_set_yf[index * batch_size: (index + 1) * batch_size],
            yd_s :train_s_set_yd[index * batch_size: (index + 1) * batch_size],
            x_t  :train_t_set_x[index * batch_size: (index + 1) * batch_size],
            yd_t :train_t_set_yd[index * batch_size: (index + 1) * batch_size]
    })
cost_d_s = theano.function(
    [index],
    Ld_s,#classification_branch[-1].errors(p_y_given_x,yf_s),
    givens={
            x_s  :train_s_set_x[index * batch_size: (index + 1) * batch_size],
            yf_s :train_s_set_yf[index * batch_size: (index + 1) * batch_size],
            yd_s :train_s_set_yd[index * batch_size: (index + 1) * batch_size],
            x_t  :train_t_set_x[index * batch_size: (index + 1) * batch_size],
            yd_t :train_t_set_yd[index * batch_size: (index + 1) * batch_size]
    })

cost_d_t = theano.function(
    [index],
    Ld_t,#classification_branch[-1].errors(p_y_given_x,yf_s),
    givens={
            x_s  :train_s_set_x[index * batch_size: (index + 1) * batch_size],
            yf_s :train_s_set_yf[index * batch_size: (index + 1) * batch_size],
            yd_s :train_s_set_yd[index * batch_size: (index + 1) * batch_size],
            x_t  :train_t_set_x[index * batch_size: (index + 1) * batch_size],
            yd_t :train_t_set_yd[index * batch_size: (index + 1) * batch_size]
    })
cost_total = theano.function(
    [index],
    ccost,#classification_branch[-1].errors(p_y_given_x,yf_s),
    givens={
            x_s  :train_s_set_x[index * batch_size: (index + 1) * batch_size],
            yf_s :train_s_set_yf[index * batch_size: (index + 1) * batch_size],
            yd_s :train_s_set_yd[index * batch_size: (index + 1) * batch_size],
            x_t  :train_t_set_x[index * batch_size: (index + 1) * batch_size],
            yd_t :train_t_set_yd[index * batch_size: (index + 1) * batch_size]
    })'''
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
#pdb.set_trace()
best_validation_loss = numpy.inf
test_score = 0.
start_time = time.clock()

done_looping = False
epoch = 0
while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):
        p = epoch - 1
        minibatch_avg_cost = train_model(minibatch_index,p)
        #cfs = cost_f_s(minibatch)
        #cds = cost_d_s(minibatch)
        #cdt = cost_d_t(minibatch_index)
        #c_total = cost_total(minibatch_index)
        #print cfs
        #print cds
        #print cdt
        #print c_total 
        #import pdb
        #pdb.set_trace()
        # iteration number
        
        iter = (epoch - 1) * n_train_batches + minibatch_index
        '''
        if (iter + 1) % validation_frequency == 0:
            #pdb.set_trace()
            # compute zero-one loss on validation set
            ##validation_losses = [validate_model(i,p)
            ##                     for i in xrange(n_valid_batches)]
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
            #import pdb
            #pdb.set_trace()
            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                # test it on the test set

                test_losses = [test_model(i,p)
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
                )'''

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
