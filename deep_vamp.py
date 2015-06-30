from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import logging
import warnings
import argparse
import os

#from make_predictions import load_dataset
import numpy as np
from os.path import isfile, join
import PIL
from random import shuffle
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far SVHN is "
            "only supported with PyTables")
import numpy
import scipy

def load_dataset(path_testset):
    dir_list = [f for f in os.listdir(path_testset) if isfile(join(path_testset,f)) and '.png' in f and '_gt' not in f] #CHANGE THIS LINE ACCORDING TO THE DATASET FILE NAMES
    rng = np.random.RandomState(seed=1234)
    rng.shuffle(dir_list)
    from PIL import Image
    images = []
    names = []      
    
    for f in dir_list:
        img = Image.open(join(path_testset,f))
        if img.size[0] > img.size[1]:
            img = img.resize((320,240),PIL.Image.ANTIALIAS) # the resize shape is (width,height)
        elif img.size[1] > img.size[0]:
            continue#    img = img.resize((240,320),PIL.Image.ANTIALIAS)

        img_npy = np.array(img,dtype='float32')
        #img_npy = img_npy.flatten()
        names.append(f)
        images.append(img_npy)
    images = np.array(images)
    return (images,names)

def generate_neg_crops(path_dataset,path_result,crop_size,nb):
    assert path_result[-1]=='/'
    images , names = load_dataset(path_dataset)
    assert nb> len(images)
    nb_crops_perImage = nb/len(images)    
    from numpy.random import randint
    for img,name in zip(images,names):
      for id in range(nb_crops_perImage):
        r = randint(0,img.shape[0]-crop_size[0]-1) # randint samples an int from [a,b]
        c = randint(0,img.shape[1]-crop_size[1]-1)
        crop = img[r:r+crop_size[0],c:c+crop_size[1],:]
        name = name.replace('.png','')
        crop_name = path_result+'Neg_'+name+'_'+str(r)+'_'+str(c)+'.png' 
        scipy.misc.imsave(crop_name, crop)



def create_dataset(DATASET, path_or_file_obj):
    _save(DATASET,path_or_file_obj)

def _save(D,f):
    #names = list(sorted(self.brains.iterkeys()))
 
    h5file = tables.openFile(f, mode="w", title="VAMP Dataset")
    filters = tables.Filters(complib='blosc')
    d_group = h5file.createGroup(h5file.root, "dataset")
    #atom_name = tables.StringAtom(64)
    atom_images = tables.Float32Atom()
    atom_labels = tables.UInt8Atom()


    node_images_array = h5file.createCArray(d_group, 'images', atom=atom_images,
                        shape=reversed((D.X.shape[1],D.X.shape[0])),
                        title="images", filters=filters)
    node_labels_array = h5file.createCArray(d_group, 'labels', atom=atom_labels,
                        shape=reversed((D.y.shape[1],D.y.shape[0])),
                        title="labels", filters=filters)

    '''
    for idx_image in range(D.X.shape[0]):
        node_images_array[...,idx_image] = D.X[idx_image,...]
        h5file.flush()

    for idx_label in range(D.y.shape[0]):
        node_labels_array[...,idx_label] = D.y[idx_label,...]
        h5file.flush()
    '''
    node_images_array[0:,0:] = D.X
    node_labels_array[0:,0:] = D.y
    h5file.flush()     
    
    h5file.close()




def toronto_preprocessing(X):
    halfdata_size = X.shape[0]/2
    X[:halfdata_size,:] = X[:halfdata_size,:] / 255.
    X[halfdata_size:,:] = X[halfdata_size:,:] / 255.
    M = X.mean(axis=0)
    X[:halfdata_size,:] = X[:halfdata_size,:] - M
    X[halfdata_size:,:] = X[halfdata_size:,:] - M
    #X = X - X[:10000].mean(axis=0)
    return X

class VAMP(DenseDesignMatrix):
    """
    TODO
    """
    def __init__(self,
                 path_dataset=None,
                 center=False,
                 gcn=None,
                 toronto_prepro=False,
                 axes=('b', 0, 1, 'c'),
                 start=None,
                 stop=None,
                 image_resize=[32,32],
                 read=None):

        self.__dict__.update(locals())
        del self.self

        # load data

        if read is None:
            assert path_dataset is not None
            base_dir = path_dataset
            #base_dir = '/home/local/USHERBROOKE/havm2701/data/Deep_VAMP/Virtual_Crops'
            dir_list = [ f for f in os.listdir(base_dir) if isfile(join(base_dir,f)) ]
            rng = np.random.RandomState(seed=1234)
            rng.shuffle(dir_list)
            from PIL import Image
            data_pair = []
            
       
            for f in dir_list:
               if ('.png' or '.jpg') not in f:
                   continue  
               if 'Neg' in f:
                   label = [1,0]
               elif 'Pos' in f:   
                   label = [0,1]
              
               elif 'FP' in f:
                   label = [1,0]
               else:
                   continue    
     
               #labels.append(label)
               img = Image.open(join(base_dir,f))
               img = img.resize((image_resize[1],image_resize[0]),PIL.Image.ANTIALIAS)
               img_npy = np.array(img,dtype='float32')
               img_npy = img_npy.flatten()
               data_pair.append((img_npy,label))

               #Add blur to Pos crops and add them to the dataset
               if label==[0,1]:
                    img = img.resize((image_resize[1]/2,image_resize[0]/2),PIL.Image.ANTIALIAS)
                    img = img.resize((image_resize[1],image_resize[0]),PIL.Image.ANTIALIAS)
                    img_npy = np.array(img,dtype='float32')
                    img_npy = img_npy.flatten()
                    data_pair.append((img_npy,label)) 
    
            rng.shuffle(data_pair)
            X = np.zeros((len(data_pair),image_resize[0]*image_resize[1]*3),dtype='float32')
            y = np.zeros((len(data_pair),2))
            for idx in range(len(data_pair)):
                X[idx,:] = data_pair[idx][0]
                y[idx,:] = data_pair[idx][1]
      
            assert X.max() == 255.
            assert X.min() == 0.
            self.center = center

            #if center:
            #    X -= 127.5

            if toronto_prepro:
                assert not center
                assert not gcn
                X = toronto_preprocessing(X)
            self.toronto_prepro = toronto_prepro

            self.gcn = gcn
            gcn = None # the following code on gcn must be fixed (axis not consistant with data)
            if gcn is not None:
                assert isinstance(gcn, float)
                X = (X.T - X.mean(axis=1)).T
                X = (X.T / np.sqrt(np.square(X).sum(axis=1))).T
                X *= gcn

            if start is not None:
                # This needs to come after the prepro so that it doesn't change
                # the pixel means computed above
                assert start >= 0
                assert stop > start
                assert stop <= X.shape[0]
                X = X[start:stop, :]
                y = y[start:stop]
                #
                assert X.shape[0] == y.shape[0]
                #X = X.swapaxes(0,1)

        elif read is not None:
            table_path = read
            h5file = tables.openFile(table_path, mode = 'r')
            image_T =  h5file.getNode('/dataset/images')
            labels_T =  h5file.getNode('/dataset/labels')
            X = image_T.read(start=start,stop=stop)
            y = labels_T.read(start=start,stop=stop)
            assert X.shape[0]==y.shape[0]


        super(VAMP, self).__init__(X=X, y=y)




class real_VAMP(DenseDesignMatrix):
    """
    TODO
    """
    def __init__(self,
                 center=False,
                 gcn=None,
                 toronto_prepro=False,
                 axes=('b', 0, 1, 'c'),
                 start=None,
                 stop=None,
                 image_resize=[32,32]):

        self.__dict__.update(locals())
        del self.self

        # load data
        #path_dataset = preprocess(path_dataset)
        base_dir = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/train_real_virtual'
        dir_list = [ f for f in os.listdir(base_dir) if isfile(join(base_dir,f)) ]
        rng = np.random.RandomState(seed=1234)
        rng.shuffle(dir_list)
        from PIL import Image
        images = []
        labels = []
        names = []
        for f in dir_list:  
            if 'Neg' in f:
                label = [1,0]
            elif 'Pos' in f:
                label = [0,1]    
            elif 'FP' in f:
                label = [1,0]
            else:
                continue

            labels.append(label)
            img = Image.open(join(base_dir,f))
            img = img.resize((image_resize[1],image_resize[0]),PIL.Image.ANTIALIAS)
            img_npy = np.array(img)
            #if img_npy.shape[2] == 3:
            #    img_npy = img_npy.swapaxes(0,2)
            img_npy = img_npy.flatten()
            names.append(f)
            images.append(img_npy)          
        Obj = {}
        Obj['data'] = images
        Obj['labels'] = labels
        Obj['names'] = names
        self.nb_samples = 0

        X = Obj['data']
        X = numpy.asarray(X,dtype='float32')
        y = numpy.asarray(Obj['labels'])
        #y = y.reshape(-1,2)
        self.names = names
        assert X.max() == 255.
        assert X.min() == 0.
        self.center = center

        if center:
            X -= 127.5

        if toronto_prepro:
            assert not center
            assert not gcn
            #if which_set == 'test':
            #    raise NotImplementedError("Need to subtract the mean of the "
            #                              "*training* set.")
            X = X / 255.
            X = X - X.mean(axis=0)
        self.toronto_prepro = toronto_prepro

        self.gcn = gcn
        gcn = None # the following code on gcn must be fixed (axis not consistant with data)
        if gcn is not None:
            assert isinstance(gcn, float)
            X = (X.T - X.mean(axis=1)).T
            X = (X.T / np.sqrt(np.square(X).sum(axis=1))).T
            X *= gcn

        if start is not None:
            # This needs to come after the prepro so that it doesn't change
            # the pixel means computed above
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop]
            assert X.shape[0] == y.shape[0]
            #X = X.swapaxes(0,1)

        super(real_VAMP, self).__init__(X=X, y=y)
