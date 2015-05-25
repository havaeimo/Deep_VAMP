
import os
import logging
import warnings
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far SVHN is "
            "only supported with PyTables")
import numpy
from theano import config
from pylearn2.datasets.dataset import Dataset
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils.iteration import SequentialSubsetIterator, resolve_iterator_class
from itertools import izip
#from lisa_brats.brains import BrainSet
import pickle
import pdb
class VAMP(Dataset):
    """
    TODO
    """
    def __init__(self,
                 path_dataset,
                 center=False,
                 gcn=None,
                 toronto_prepro=False,
                 axes=('b', 0, 1, 'c'),
                 num_minibatches_train=None,
                 num_minibatches_test=None,
                 start=None,
                 stop=None):

        self.__dict__.update(locals())
        del self.self

        # load data
        path_dataset = preprocess(path_dataset)

        self.nb_samples = 0
        p_dataset = open( path_dataset, "rb" )
        Obj = pickle.load(p_dataset)
        pdb.set_trace()
        X = Obj['data']
        X = numpy.asarray(X,dtype='float32')
        y = numpy.asarray(Obj['labels'])
        assert X.max() == 255.
        assert X.min() == 0.
        self.center = center

        if center:
            X -= 127.5

        if toronto_prepro:
            assert not center
            assert not gcn
            if which_set == 'test':
                raise NotImplementedError("Need to subtract the mean of the "
                                          "*training* set.")
            X = X / 255.
            X = X - X.mean(axis=0)
        self.toronto_prepro = toronto_prepro


        self.gcn = gcn
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

        self.current_index_training = 0
        self.current_index_testing = 0

        self.axes = axes

        super(VAMP, self).__init__()


    def iterator(self, mode=None, batch_size=None, num_batches=None,
                topo=None, targets=None, rng=None, return_tuple=True,
                data_specs=None):
        """
        method inherited from Dataset
        """
        
        """if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        if rng is not None:
            mode = 'random_slice'"""
        ''
        if batch_size != 128:
            raise ValueError("only a batch_size of 128 is supported")
        self.count = 0
        if rng is None:
            self.stochastic = False
            if self.num_minibatches_test is None:
                raise ValueError("not able to be used for testing: num_minibatches_test not set")
        else:
            self.stochastic = True
            if self.num_minibatches_train is None:
                raise ValueError("not able to be used for training: num_minibatches_train not set")
        return self


    def __iter__(self):
        return self

    def next(self):
        if ( self.stochastic and self.count >= self.num_minibatches_train) or ((not (self.stochastic)) and self.count >= self.num_minibatches_test):
            raise StopIteration()

        X0 = numpy.zeros(( 3, 64, 32, 128), dtype=numpy.float32)
        y = numpy.zeros((128,1), dtype=numpy.float32)
        index = 0
        if self.stochastic:
            current_index_array = self.current_index_training
        else:
            current_index_array = self.current_index_testing

        num_examples = 128
        current_index_array
        data = self.X[current_index_array:current_index_array+128,:,:,:]
        labels = self.y[current_index_array:current_index_array+128]
        #ipdb.set_trace()
        current_index_array += num_examples
        assert len(data) <= num_examples
        if len(data) < num_examples:
                current_index_array = num_examples-len(data)
                data = numpy.concatenate((data, self.X[0:num_examples-len(data),:,:,:]), axis=0)
                labels = numpy.concatenate((labels, self.y[0:num_examples-len(data)]), axis=0)
                assert len(data) == num_examples
                for d in data:
                    #ipdb.set_trace()
                    X0[..., index] = data.swapaxes(0,3)
                    y[index] = labels
                    index += 1
        
        
        mini_batch = (X0, y)
        
        assert index == 128
        self.count += 1
        if self.stochastic:
            self.current_index_training = current_index_array
        else:
            self.current_index_testing = current_index_array
        return mini_batch

        
    @property
    def batch_size(self):
        """
        .. todo::
            WRITEME
        """
        return 128

    @property
    def num_batches(self):
        """
        .. todo::
            WRITEME
        """
        if self.stochastic:
            return self.num_minibatches_train
        else:
            return self.num_minibatches_test

    @property
    def num_examples(self):
        """
        .. todo::
            WRITEME
        """
        if self.stochastic:
            return self.num_minibatches_train * 128
        else:
            return self.num_minibatches_test * 128

    @property
    def uneven(self):
        """
        .. todo::
            WRITEME
        """
        return False 

D = VAMP(path_dataset='/home/local/USHERBROOKE/havm2701/scripts/Deep_VAMP/vamp_virtual.pkl',num_minibatches_train=1000)

D.iterator(batch_size=128,rng=True)

t = D.__iter__()
next(t)
