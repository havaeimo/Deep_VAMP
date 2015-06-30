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

def resize_ground_truth(im_path,txt_path,im_size=(240,320)):
    target_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Test/Annotations'
    result_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Test/Posframes_Annotations'

    list_dir = [d for d in os.listdir(txt_path) if isfile(join(txt_path,d)) and '.txt' in d]

    for name in list_dir:


        f = open(join(txt_path,name),'r')
        content = f.readlines()
        f.close()
        g = open(join(txt_path,name),'w')

        content_n = []
        for line in content:
            line = line.replace('PEDESTRIAN-OBLIGATORY','')
            line= 'PEDESTRIAN-OBLIGATORY' + ' ' + line
            g.write(line)
        #content = content.replace('PEDESTRIAN-OBLIGATORY','')
        #d_name = 'CropsPos_' + name
        #d_name = old_name.replace('.png','')
        
        g.close()
        #command = 'cp ' + target_name + ' ' + result_path
        #print command
        #os.system(command)



def generate_bb_score_file_from_maps(path,scales,identifier,threshold):

    dir_list = [f.replace('.png','') for f in os.listdir(path) if '.png' in f]
    dir_list = [g.replace(g.split('_')[-1],'') for g in dir_list]
    dir_list = set(dir_list)
    #pdb.set_trace()
    for f in dir_list:
        for scale in scales:
            name = join(path,f + str(scale) + '.png')
            if isfile(name):
                img = Image.open(name)
                img_npy = np.array(img,dtype='float32')
                img_npy /= 255.0


                img_npy_big = np.zeros((img_npy.shape[0]+128,img_npy.shape[1]+64),dtype=np.float32)
                img_npy_big[:img_npy.shape[0],:img_npy.shape[1]] = img_npy
                img_npy = img_npy_big
                #pos_map_shape = [(scale * 240) -128 , (scale * 320) -64 ]
                reference_map_size = [240,320]
                reverse_scale = [(reference_map_size[0]*1.0)/(img_npy.shape[0]*1.0),(reference_map_size[1]*1.0)/(img_npy.shape[1]*1.0)]
                img_npy = scipy.ndimage.zoom(img_npy, reverse_scale, order=3)
                bounding_box_size = ((1/scale)*128,(1/scale)*64)
                MultiScaleResultImage = [(img_npy,bounding_box_size)]
                name_text = f[:-1]
                
                
                generate_BoundingBOx_score_file(MultiScaleResultImage,name_text,path,threshold=0.01)


def generate_BoundingBOx_score_file(MultiScaleResultImage,name,save_path,threshold=0):
    name = name.replace('.png','')
    save_path = join(save_path,'txt_results')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f = open(join(save_path,name+'.txt'),'a')
    texts = ''    
    for (map,bounding_box_size) in MultiScaleResultImage:
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if map[i,j]>threshold:
                    texts += str(j)+' '+str(i)+' '+str(int(bounding_box_size[1]))+' '+str(int(bounding_box_size[0]))+' '+str(map[i,j])+'\n'
    f.write(texts)
    f.close()                

def makepatches(image,i0,j0):

    idx = 0
    height,width = image.shape[:-1]
    #patches = np.zeros(((height-input_shape[0])*(width-input_shape[1]),input_shape[0],input_shape[1],3),dtype=np.float32)
    patches = np.zeros((batch_size,input_shape[0],input_shape[1],3),dtype=np.float32)

    for i in range(i0,height-input_shape[0]):
        for j in range(j0,width-input_shape[1]):
            patches[idx,...] = image[i:i+input_shape[0],j:j+input_shape[1],:]
            idx += 1
            if idx==batch_size:
                break    
        if idx==batch_size:
            break

    i0=i
    j0=j        
    assert patches.shape[0] == batch_size #(height-input_shape[0])*(width-input_shape[1])        
    return (patches,i0,j0)

def generate_prediction_patchwise(data,fprop,batch_size=100):

    results = []
    result = []
    image = data[0]
    #patches = np.zeros((batch_size,input_shape[0],input_shape[1],3),dtype=np.float32)
    height,width = image.shape[:-1]
    idx=0
    patches_list=[]
    for i in range(height-input_shape[0]):
        for j in range(width-input_shape[1]):
            patches_list.append(image[i:i+input_shape[0],j:j+input_shape[1],:])

            if idx==batch_size-1 or (i ==(height-input_shape[0]-1) and j==(width-input_shape[1]-1)):            
                patches = np.asarray(patches_list)     
                result_patches = generate_prediction(patches,fprop,batch_size=100)   
                result_patches = np.array(result_patches).reshape(len(result_patches),2)
                result.extend(result_patches)
                patches_list=[]    
                idx=0  

            idx += 1

    print j          
    result = np.asarray(result)               
    result_image = result.reshape(height-input_shape[0],width-input_shape[1],2)
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
    dir_list = [f for f in os.listdir(path_testset) if isfile(join(path_testset,f)) and '.png' in f and '_gt' not in f] #CHANGE THIS LINE ACCORDING TO THE DATASET FILE NAMES
    rng = np.random.RandomState(seed=1234)
    rng.shuffle(dir_list)
    from PIL import Image
    images = []
    names = []      
    
    for f in dir_list:
        img = Image.open(join(path_testset,f))
        #if img.size[1] > img.size[0]:
        #    img = img.resize((320,240),PIL.Image.ANTIALIAS) # the resize shape is (width,height)
        #elif img.size[0] > img.size[1]:
        #    continue#    img = img.resize((240,320),PIL.Image.ANTIALIAS)
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
    
    parser.add_argument('path_testset', type=str,
                        help='The path to test images.')
    parser.add_argument('result_path', type=str,
                        help='The path to save results')
    #parser.add_argument('patch_shape', type=int,
    #                    help='The size of the input patch window.'),
    #parser.add_argument('label_patch_shape', type=int,
    #                    help='The size of the predicted patch window.'),
    #parser.add_argument('num_channels', type=int,
    #                    help='Number of channels in the dataset.'),
    args = parser.parse_args()
    #path_testset = self.testset
    #path_testset = '/home/local/USHERBROOKE/havm2701/data/DBFrames'

    #result_path = '/home/local/USHERBROOKE/havm2701/git.repos/Deep_VAMP/ConvRecL/test_results2/'
    path_testset = args.path_testset
    result_path = args.result_path

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model = cPickle.load(args.model)
    scales = [0.65,0.75,1]#,1.5,2,]
    threshold = 0.01
    batch_size=100
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


    #testdata = testdata[:500,...]
    #name_testdata = name_testdata[:500]


    #if os.path.exists(fname):
    #    print fname + ' exists already. skipping'
    #    continue
    #prediction = generate_prediction(testdata, fprop)
    testdata = toronto_preprocessing(testdata)
    #prediction = generate_prediction_patchwise(testdata,fprop)
    ii = 0

    for name, test_image in izip(name_testdata,testdata):
        if isfile(join(result_path+name.replace('.png','')+'.txt')):
            print name + ' exists'
            ii +=1    
            continue
        
        for scale in scales:
            
            scaled_image = scipy.ndimage.zoom(test_image, [scale,scale,1], order=3).copy()
            prob_map = generate_prediction_patchwise([scaled_image],fprop)

            prob_map = prob_map[0]
            pos_map = prob_map[...,1]
           
            pos_map_big = np.zeros((pos_map.shape[0]+128,pos_map.shape[1]+64),dtype=np.float32)
            pos_map_big[:pos_map.shape[0],:pos_map.shape[1]] = pos_map
            pos_map = pos_map_big
            
            reference_map_size = [240,320]
            reverse_scale = [(reference_map_size[0]*1.0)/(pos_map.shape[0]*1.0),(reference_map_size[1]*1.0)/(pos_map.shape[1]*1.0)]
            
            pos_map = scipy.ndimage.zoom(pos_map, reverse_scale, order=3)
            bounding_box_size = ((1/scale)*input_shape[0],(1/scale)*input_shape[1])
            
            MultiScaleResultImage = [(pos_map,bounding_box_size)]
            #neg_map = prob_map[...,0]
            #name = name.replace('.png','')
            pos_name = join(result_path,name.replace('.png','')+'_'+str(scale)+'.png')


            #neg_name = join(result_path,name+'_neg.png')
            image_name = join(result_path,name)        
            scipy.misc.imsave(pos_name, pos_map)
            #scipy.misc.imsave(neg_name, neg_map)
            scipy.misc.imsave(image_name, test_image)
            print name+'_'+str(scale)+'>> image_shape '+str(scaled_image.shape) +'>> '+str(ii+1)+' of '+str(len(testdata))
            generate_BoundingBOx_score_file(MultiScaleResultImage,name,result_path,threshold)

        ii+=1         
            #fhandle = open(fname, 'wb+')
            #numpy.save(fhandle, prediction)
            #fhandle.close()
