from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import logging
import warnings
import argparse
import pdb
import os
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

def toronto_preprocessing(X):
      X = X / 255.
      X = X - X.mean(axis=0)
      return X

class VAMP(DenseDesignMatrix):
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
        base_dir = '/home/local/USHERBROOKE/havm2701/data/Deep_VAMP/Virtual_Crops'
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
            img_npy = np.array(img,dtype='float32')
            self.nb_channels = img_npy.shape[-1]
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
        #pdb.set_trace()
        #y = y.reshape(-1,2)

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

        super(VAMP, self).__init__(X=X, y=y)

    def get_reshaped_images(self):
        self.X = self.X.reshape(-1,self.image_resize[0],self.image_resize[1],self.nb_channels)
        #self.y = self.y 
        return self
        
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
        base_dir = '/home/local/USHERBROOKE/havm2701/data/Deep_VAMP/Crops'
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
        #pdb.set_trace()
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
            #pdb.set_trace()
            assert X.shape[0] == y.shape[0]
            #X = X.swapaxes(0,1)

        super(real_VAMP, self).__init__(X=X, y=y)
