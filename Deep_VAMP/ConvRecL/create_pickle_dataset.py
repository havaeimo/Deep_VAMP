from pylearn2.datasets.deep_vamp import toronto_preprocessing
from pylearn2.datasets.deep_vamp import VAMP
from pylearn2.datasets.deep_vamp import generate_neg_crops
from pylearn2.datasets.deep_vamp import create_dataset

data_set = VAMP(
            path_dataset='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Train/crops_train_pos_neg',
            center=False,
            gcn=False,
            toronto_prepro=True,
            #axes: ['c', 0, 1, 'b'],
            start=0,
            stop=20282,
            image_resize= [128,64],
            #read = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/TEST'
            )
#import pdb
#pdb.set_trace()
save_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/PN_REAL'
create_dataset(data_set,save_path)
'''

#path_dataset: '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/train_real_virtual',
#      center: False,
#      gcn: False,
#      toronto_prepro: True,
#      #axes: ['b', 0, 1, 'c'],
#      start: 0,
#      stop: 20000,
#      image_resize: [128,64]
#    },

path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Train/FramesNeg'
result_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Train/crops_train_Neg/'

generate_neg_crops(path,result_path,[128,64],10000)

data_set_train = VAMP(
            #path_dataset='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/train_real_pn_virtual/',
            center=False,
            gcn=False,
            toronto_prepro=True,
            #axes: ['c', 0, 1, 'b'],
            start=0,
            stop=25000,
            image_resize= [128,64],
            read = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/TRAIN_REAL_PN_VIRTUAL'
            )
save_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/TRAIN_REAL_PN_VIRTUAL'
#create_dataset(data_set,save_path)

data_set_valid= VAMP(
            #path_dataset='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/train_real_pn_virtual/',
            center=False,
            gcn=False,
            toronto_prepro=True,
            #axes: ['c', 0, 1, 'b'],
            start=25000,
            stop=29000,
            image_resize= [128,64],
            read = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/TRAIN_REAL_PN_VIRTUAL'
            )
data_set_test = VAMP(
            #path_dataset='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/train_real_pn_virtual/',
            center=False,
            gcn=False,
            toronto_prepro=True,
            #axes: ['c', 0, 1, 'b'],
            start=29000,
            stop=31000,
            image_resize= [128,64],
            read = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/TRAIN_REAL_PN_VIRTUAL'
            )
import pdb
pdb.set_trace()
'''
