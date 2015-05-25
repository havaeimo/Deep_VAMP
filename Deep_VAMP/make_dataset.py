

import argparse
import pdb
import os
import numpy as np
from os.path import isfile, join
import PIL
from random import shuffle
import pickle

def make_argument_parser():
    parser = argparse.ArgumentParser(description='Load data and dump it '
                                                 'to a NumPy zip archive, '
                                                 'loadable by the Vampset '
                                                 'class.')
    parser.add_argument('-B', '--base-dir', type=str, default='.',
                        help='Directory containing data '
                             '(default: current directory)')
    parser.add_argument('-R', '--result-dir', type=str, default='.',
                        help='Directory containing data '
                             '(default: current directory)')
    #parser.add_argument('-O', '--train-out', type=str,
    #                    required=True,
    #                    help='Filename to use for the serialized train set.')

    #parser.add_argument('-V', '--valid-out', type=argparse.FileType('wb'),
    #                    required=False,
    #                    help='Filename to use for the serialized valid set.')

    #parser.add_argument('-T', '--test-out', type=argparse.FileType('wb'),
    #                   required=False,
    #                  help='Filename to use for the serialized test set.')
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    # Remove trailing slashes.

    base_dir = args.base_dir
    result_path = args.result_dir
    dir_list = [ f for f in os.listdir(base_dir) if isfile(join(base_dir,f)) ]
    pdb.set_trace()
    shuffle(dir_list)
    from PIL import Image
    images = []
    labels = []
    names = []
    for f in dir_list:  
        if 'Neg' in f:
            label = [1,0]
        if 'Pos' in f:
            label = [0,1]
        else:
            continue
        labels.append(label)
        img = Image.open(join(base_dir,f))
        img = img.resize((32,32),PIL.Image.ANTIALIAS)
        img_npy = np.array(img)
        if img_npy.shape[2] == 3:
            img_npy = img_npy.swapaxes(0,2)
        img_npy = img_npy.flatten()
        names.append(f)
        images.append(img_npy)
            
    Obj = {}
    Obj['data'] = images
    Obj['labels'] = labels
    Obj['names'] = names
    pdb.set_trace()

    with open(result_path, 'wb') as output:
        pickle.dump(Obj, output, pickle.HIGHEST_PROTOCOL)
