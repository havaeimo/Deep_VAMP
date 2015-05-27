import models
import deepvamp_dataset

class DeepVamp(object):

    def __init__(self, dataset, random_seed, hiddensize=[500,100]):
    	input = T.matrix('input')   # the data is presented as rasterized images
    	target = T.matrix('target')  # the labels are presented as 1D vector of

    	input_size = output_size = dataset['input_size']
    	self.rng = np.random.mtrand.RandomState(random_seed)

    	self.layers = [medels.LeNetConvPoolLayer(self.rng, input, filter_shape=(?,?), image_shape=(?,?), activation=None, pool_size=(?, ?), pool_stride=(?,?))]

    	self.layers += models.OutputLayer(input=self.layers[-1].out, filter_shape=(?,?), image_shape=(?,?),n_classes=?)

    	cost_obj = models.Cost(self.layers[-1].out, target)
    	cost = cost_obj.out

        layer_parameters = []
        for i in range(2):
            layer_parameters += self.layers[i].params

        parameters_gradiants = T.grad(cost, layer_parameters)    
        updates = []
        for param, param_gradiant in zip(layer_parameters, parameters_gradiants):
            updates += (param, param - self.learning_rate * param_gradiant)


        #self.train = theano.function([input],cost, updates=updates)    


        self.train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


        index = T.lscalar()
        self.learn = theano.function(name='learn',
                                     inputs=[index],
                                     outputs=loss,
                                     updates=updates,
                                     givens={input: dataset['train']['data'][index * batch_size:(index + 1) * batch_size], target: dataset['train']['target'][index * batch_size:(index + 1) * batch_size]}) 



        self.use = theano.function(name='use',
                                   inputs=[input],
                                   outputs=output) 

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

