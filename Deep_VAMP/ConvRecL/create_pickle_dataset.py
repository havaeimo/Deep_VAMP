from pylearn2.datasets.deep_vamp import toronto_preprocessing
from pylearn2.datasets.deep_vamp import VAMP
from pylearn2.datasets.deep_vamp import generate_neg_crops
'''
data_set = VAMP(
            path_dataset='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/train_real_virtual/',
            center=False,
            gcn=False,
            toronto_prepro=True,
            #axes: ['c', 0, 1, 'b'],
            start=0,
            stop=4000,
            image_resize= [128,64],
            read = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/train_real_virtual.pkl'
            )
import pdb
pdb.set_trace()
save_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/train_real_virtual.pkl'
create_dataset(data_set,save_path)


#path_dataset: '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/train_real_virtual',
#      center: False,
#      gcn: False,
#      toronto_prepro: True,
#      #axes: ['b', 0, 1, 'c'],
#      start: 0,
#      stop: 20000,
#      image_resize: [128,64]
#    },
'''
path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Train/FramesNeg'
result_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Train/crops_train_Neg/'

generate_neg_crops(path,result_path,[128,64],4000)


