import numpy
import models
from models import relu
import argparse
import pdb
import os
import theano.tensor as T
import numpy as np
import theano
import time

def train(classifier,max_epochs):

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
    n_epochs = max_epochs
    epoch = 0
    done_looping = False
   
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(classifier.n_train_batches):
            pdb.set_trace()
            minibatch_avg_cost = classifier.train(minibatch_index)
            # iteration number

            iter = (epoch - 1) * classifier.n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [classifier.valid(i) for i
                                     in xrange(classifier.n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%, training error last minibatch of epoch %f %%' %
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