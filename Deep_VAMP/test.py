from vamp_dataset_copy import VAMP
import pdb
D = VAMP(path_dataset = 'vamp_virtual.pkl', center=False, gcn=False, toronto_prepro=True, axes=('c', 0, 1, 'b'), start=0, stop=100) 
import scipy
scipy.misc.imsave('outfile3.jpg', D.X[...,0])


print D.X
pdb.set_trace()
