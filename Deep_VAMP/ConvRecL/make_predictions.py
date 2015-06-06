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
import scipy.ndimage
def generate_BoundingBOx_score_file(MultiScaleResultImage,name,threshold=0):
    f = open(join(result_path+name+'.txt'),'w')
    texts = ''    
    for (map,bounding_box_size) in MultiScaleResultImage:
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if map[i,j]>threshold:
                    texts += str(i)+' '+str(j)+' '+str(int(bounding_box_size[1]))+' '+str(int(bounding_box_size[0]))+' '+str(map[i,j])+'\n'
    f.write(texts)                

def makepatches(image):

    idx = 0
    height,width = image.shape[:-1]
    patches = np.zeros(((height-input_shape[0])*(width-input_shape[1]),input_shape[0],input_shape[1],3),dtype=np.float32)
    for i in range(height-input_shape[0]):
        for j in range(width-input_shape[1]):
            patches[idx,...] = image[i:i+input_shape[0],j:j+input_shape[1],:]
            idx += 1
    assert patches.shape[0] == (height-input_shape[0])*(width-input_shape[1])        
    return patches

def generate_prediction_patchwise(data,fprop,batch_size=100):

    results = []
    for image in data:
        height,width = image.shape[:-1]
        image_patches = makepatches(image)
        result_patches = generate_prediction(image_patches,fprop,batch_size=100)
        result_patches = np.array(result_patches).reshape(len(result_patches),2)
        result_image = result_patches.reshape(height-input_shape[0],width-input_shape[1],2)

        results.append(result_image)

    return results

def prepare_batch(batch,axes,batch_size=100):

    if axes == ('c',0,1,'b'):
        batch = batch.swapaxes(0, 3).copy()
        num_samples = batch.shape[3]      
        if num_samples < batch_size:
            buffer_batch = np.zeros((batch.shape[0],batch.shape[1],batch.shape[2],batch_size),dtype=np.float32)
            buffer_batch[:,:,:,0:num_samples] = batch
            batch = buffer_batch
    elif axes == ('b',0,1,'c'):
        num_samples = batch.shape[0]      
        if num_samples < batch_size:
            buffer_batch = np.zeros((batch_size,batch.shape[1],batch.shape[2],batch.shape[3]),dtype=np.float32)
            buffer_batch[0:num_samples,:,:,:] = batch
            batch = buffer_batch


    return (batch,num_samples)

def generate_prediction(data,fprop,batch_size=100):
    axes = model.input_space.axes
    batches = int(numpy.ceil(data.shape[0] / float(batch_size)))
    results = []
    for b in xrange(batches):
        batch = data[b * batch_size:(b + 1) * batch_size]
        #batch = batch.swapaxes(0, 3).copy()
        batch,num_samples = prepare_batch(batch,axes,batch_size=100)
        results_batch = fprop(batch)
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

def make_multiple_scales(image, scales):
    #scaled_size = [(320*scale,240*scale)  for scale in scales]
    scaled_image=[]
    for scale in scales:
        scaled_image += [scipy.ndimage.zoom(image, [scale,scale,1], order=3)]
    return scaled_image    

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
    scales = [0.75,1]
    threshold = 0.01 
    #model.layers[0].input_space.shape = (240,320)
    #model.layers[0].desired_space.shape = (240, 320)   
    
    X = model.get_input_space().make_theano_batch()
    fprop = theano.function([X], model.fprop(X))
    input_shape = model.input_space.shape
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

        MultiScaleResultImage = []
        for scale in scales:
            scaled_image = scipy.ndimage.zoom(test_image, [scale,scale,1], order=3).copy()
            prob_map = generate_prediction_patchwise([scaled_image],fprop)

            prob_map = prob_map[0]
            pos_map = prob_map[...,1]
            
            reference_map_size = [240-input_shape[0],320-input_shape[1]]
            reverse_scale = [(reference_map_size[0]*1.0)/(pos_map.shape[0]*1.0),(reference_map_size[1]*1.0)/(pos_map.shape[1]*1.0)]
            pos_map = scipy.ndimage.zoom(pos_map, reverse_scale, order=3)
            bounding_box_size = (reverse_scale[0]*input_shape[0],reverse_scale[1]*input_shape[1])
            
            MultiScaleResultImage += [(pos_map,bounding_box_size)]
            #neg_map = prob_map[...,0]
            pos_name = join(result_path,name+'_'+str(scale)+'.png')
            #neg_name = join(result_path,name+'_neg.png')
            image_name = join(result_path,name)        
            scipy.misc.imsave(pos_name, pos_map)
            #scipy.misc.imsave(neg_name, neg_map)
            scipy.misc.imsave(image_name, test_image)
            print name+'_'+str(scale)+'>> image_shape '+str(scaled_image.shape) +'>> '+str(ii+1)+' of '+str(len(testdata))
        pdb.set_trace()    
        generate_BoundingBOx_score_file(MultiScaleResultImage,name,threshold)

        ii+=1         
            #fhandle = open(fname, 'wb+')
            #numpy.save(fhandle, prediction)
            #fhandle.close()
