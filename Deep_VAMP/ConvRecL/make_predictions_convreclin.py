import pdb
import numpy as np
#from matplotlib.pyplot import imsave
import numpy
from os.path import isfile, join
import argparse
from itertools import izip
import scipy
import cPickle
import theano
import os
import os.path
from PIL import Image
import PIL
from pylearn2.datasets.deep_vamp import toronto_preprocessing

def makepatches(image):

    idx = 0
    height,width = image.shape[:-1]
    patches = np.zeros(((height-128)*(width-64),128,64,3),dtype=np.float32)
    for i in range(height-128):
        for j in range(width-64):
            patches[idx,...] = image[i:i+128,j:j+64,:]
            idx += 1
    assert patches.shape[0] == (height-128)*(width-64)        
    return patches

def generate_prediction_patchwise(data,fprop,batch_size=100):

    results = []
    for image in data:
        height,width = image.shape[:-1]
        image_patches = makepatches(image)
        result_patches = generate_prediction(image_patches,fprop,batch_size=100)
        result_patches = np.array(result_patches).reshape(len(result_patches),2)
        result_image = result_patches.reshape(height-128,width-64,2)

        results.append(result_image)

    return results


def generate_prediction(data,fprop,batch_size=100):
  
    batches = int(numpy.ceil(data.shape[0] / float(batch_size)))
    results = []
    for b in xrange(batches):
        batch = data[b * batch_size:(b + 1) * batch_size]
        #batch = batch.swapaxes(0, 3).copy()

        num_samples = batch.shape[0]      
        if num_samples < batch_size:
            buffer_batch = np.zeros((batch_size,batch.shape[1],batch.shape[2],batch.shape[3]),dtype=np.float32)
            buffer_batch[0:num_samples,:,:,:] = batch
            batch = buffer_batch

        results_batch = fprop(batch)
        pdb.set_trace()
        if num_samples < batch_size:
            results_batch = results_batch[0:num_samples,...]

        results.extend(results_batch)    
    return results
    
def load_dataset(path_testset):
    dir_list = [f for f in os.listdir(path_testset) if isfile(join(path_testset,f)) and '.jpg' in f and '_gt' not in f] #CHANGE THIS LINE ACCORDING TO THE DATASET FILE NAMES
    rng = np.random.RandomState(seed=1234)
    rng.shuffle(dir_list)
    from PIL import Image
    images = []
    names = []      
    
    for f in dir_list:
        img = Image.open(join(path_testset,f))

        img = img.resize((320,240),PIL.Image.ANTIALIAS) # the resize shape is (width,height)
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
    path_testset = '/home/local/USHERBROOKE/havm2701/data/DBFrames'

    result_path = '/home/local/USHERBROOKE/havm2701/git.repos/Deep_VAMP/ConvRecL/test_results2/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model = cPickle.load(args.model)
    del model.layers[-1]
    del model.layers[-1]
    #num_channels = model.input_space.num_channels
    #model.layers[0].input_space.shape = (240,320)
    #model.layers[0].desired_space.shape = (240, 320)   
    X = model.get_input_space().make_theano_batch()
    fprop = theano.function([X], model.fprop(X))
    #theano.printing.debugprint(f)
    #fprop_input_shape = model.get_input_space().shape
    testdata,name_testdata = load_dataset(path_testset)
    #testdata = testdata[0]
    #name_testdata = testdata[1]
    testdata = testdata[:500,...]
    name_testdata = name_testdata[:500]
    #if os.path.exists(fname):
    #    print fname + ' exists already. skipping'
    #    continue
    #prediction = generate_prediction(testdata, fprop)
    testdata = toronto_preprocessing(testdata)
    #prediction = generate_prediction_patchwise(testdata,fprop)
    ii = 0
    for name, test_image in izip(name_testdata,testdata):
        prob_map = generate_prediction_patchwise([test_image],fprop)
        import pdb
        pdb.set_trace()
        prob_map = prob_map[0]
        pos_map = prob_map[...,1]
        #neg_map = prob_map[...,0]
        pos_name = join(result_path,name+'_pos.png')
        #neg_name = join(result_path,name+'_neg.png')
        image_name = join(result_path,name)        
        scipy.misc.imsave(pos_name, pos_map)
        #scipy.misc.imsave(neg_name, neg_map)
        scipy.misc.imsave(image_name, test_image)
        print name+'>> '+str(ii+1)+' of '+str(len(testdata))
        ii+=1         
        #fhandle = open(fname, 'wb+')
        #numpy.save(fhandle, prediction)
        #fhandle.close()
