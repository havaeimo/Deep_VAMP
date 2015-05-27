import models


class DeepVamp(object):

    def __init__(self, dataset, random_seed, hiddensize=[500,100]):
    	input = T.matrix('input')   # the data is presented as rasterized images
    	target = T.matrix('target')  # the labels are presented as 1D vector of

    	input_size = output_size = dataset['input_size']
    	self.rng = np.random.mtrand.RandomState(random_seed)

    	self.layers = [medels.LeNetConvPoolLayer(self.rng, input, filter_shape=(?,?), image_shape=(?,?), activation=None, pool_size=(?, ?), pool_stride=(?,?))]

    	self.layers += models.OutputLayer(input=self.layers[-1].out, filter_shape=(?,?), image_shape=(?,?),n_classes=?)

    	
