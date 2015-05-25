import numpy as np
#from matplotlib.pyplot import imsave
import numpy
from os.path import isfile, join
import argparse

import cPickle
import theano
import os
import os.path
import pdb
from PIL import Image
import PIL

def generate_prediction(data,fprop,batch_size=100):

  
    batches = int(numpy.ceil(data.shape[0] / float(batch_size)))
    results = []
    for b in xrange(batches):
        batch = data[b * batch_size:(b + 1) * batch_size]
        #batch = batch.swapaxes(0, 3).copy()
        pdb.set_trace()
        num_samples = batch.shape[0]         
        if num_samples < batch_size:
            buffer_batch = np.zeros((batch_size,batch.shape[1],batch.shape[2],batch.shape[3]),dtype=np.float32)
            buffer_batch[0:num_samples,:,:,:] = batch
            batch = buffer_batch

        batch = batch.swapaxes(0, 3).copy()
        results_batch = fprop(batch)
        if num_samples < batch_size:
            results_batch = results_batch[0:num_samples,...]

        results.extend(results_batch)
    return results




    return 0
    
def load_batch(path_testset):
    dir_list = [f for f in os.listdir(path_testset) if isfile(join(path_testset,f)) and 'png' in f]
    from PIL import Image
    images = []
    names = []      

    for f in dir_list:
        img = Image.open(join(path_testset,f))

        img = img.resize((80,80),PIL.Image.ANTIALIAS) # the resize shape is (width,height)
        img_npy = np.array(img,dtype='float32')
        #img_npy = img_npy.flatten()
        names.append(f)
        images.append(img_npy)
    images = np.array(images)
    return (images,names)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate the DICE score for a BrainSet')
    parser.add_argument('model', type=argparse.FileType('r'),
                        help='A serialized pylearn2 model.')
    #####parser.add_argument('testset', type=str,
    #                    help='The path to test images.'),
    #parser.add_argument('patch_shape', type=int,
    #                    help='The size of the input patch window.'),
    #parser.add_argument('label_patch_shape', type=int,
    #                    help='The size of the predicted patch window.'),
    #parser.add_argument('num_channels', type=int,
    #                    help='Number of channels in the dataset.'),
    args = parser.parse_args()
    #path_testset = self.testset
    path_testset = '/home/local/USHERBROOKE/havm2701/data/Deep_VAMP/INRIA/Test/FramesNeg'
    model = cPickle.load(args.model)
    del model.layers[-1]
    pdb.set_trace()
    num_channels = model.input_space.num_channels
    #ipdb.set_trace() 
    model.layers[0].input_space.shape = (80, 80)
    #model.layers[0].desired_space.shape = (80, 80)
    X = model.get_input_space().make_theano_batch()
    fprop = theano.function([X], model.fprop(X))
    #theano.printing.debugprint(f)
    #fprop_input_shape = model.get_input_space().shape
    testdata = load_batch(path_testset)
    testdata = testdata[0]
    #if os.path.exists(fname):
    #    print fname + ' exists already. skipping'
    #    continue
    pdb.set_trace()
    prediction = generate_prediction(testdata, fprop)


        #fhandle = open(fname, 'wb+')
        #numpy.save(fhandle, prediction)
        #fhandle.close()
        #ipdb.set_trace()
