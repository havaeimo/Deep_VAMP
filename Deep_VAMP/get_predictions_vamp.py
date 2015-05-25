import numpy as np
import numpy
import argparse
#from lisa_brats.brains import BrainSet
from pylearn2.datasets.deep_vamp import real_VAMP
#from lisa_brats.misc.preprocessing import standardize_nonzeros
import cPickle
import theano
import os
import os.path
import pdb
from pylearn2.utils import serial
import theano.tensor as T
from os.path import isfile, join
from PIL import Image
import scipy


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='get predictions for the model from a dataset')
    parser.add_argument('model_path', type=str,
                        help='A serialized pylearn2 model.')

    result_dir = '/home/local/USHERBROOKE/havm2701/scripts/Deep_VAMP/image_T2/'
    
    args = parser.parse_args()

    dir_true_ped = join(result_dir,'pedestrain')
    if not os.path.exists(dir_true_ped):
        os.makedirs(dir_true_ped)

    dir_true_neg = join(result_dir,'neg')
    if not os.path.exists(dir_true_neg):
        os.makedirs(dir_true_neg)

    dir_error = join(result_dir,'error')
    if not os.path.exists(dir_error):
        os.makedirs(dir_error)

    model_path = args.model_path        
    model = serial.load(model_path)
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop( X )
    Y = T.argmax( Y, axis = 1 )
    f = theano.function( [X], Y )


    data_set = real_VAMP(center=False,
            gcn=False,
            toronto_prepro=True,
            #axes: ['c', 0, 1, 'b'],
            start=0,
            stop=4000,
            image_resize=32
            )

    image_resize = 32
    x_test = [d.reshape(image_resize,image_resize,3) for d in data_set.X]
    x_test = np.array(x_test, dtype='float32')
    x_test = x_test.swapaxes(0,3)
    #x_test = data_set.X
    labels = data_set.y

    y = f( x_test )

    T = []
    F = []

    names = data_set.names
    x_test = x_test.swapaxes(0,3)
    for i,y_hat in  enumerate(y):
        name = names[i]
        x = x_test[i]
        target = labels[i]

        if target[y_hat] == 1 and y_hat == 1:
    
            ped_path = join(dir_true_ped,name)
            scipy.misc.imsave(ped_path, x)
        elif target[y_hat] == 1 and y_hat == 0:

            neg_path = join(dir_true_neg,name)
            scipy.misc.imsave(neg_path, x) 
        elif target[y_hat] == 0:
            error_path = join(dir_error,name)
            scipy.misc.imsave(error_path, x)
        

