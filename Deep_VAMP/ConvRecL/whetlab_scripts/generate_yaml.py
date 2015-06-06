#import ipdb

def get_accuracy(model_path):
        from pylearn2.utils import serial
        model = serial.load(model_path)
        monitor = model.monitor
        accuracy = 1.0 - monitor.channels['valid_y_misclass'].val_record[-1]
        return accuracy

def fix_yaml_params(yaml_param):

        #ipdb.set_trace()
        yaml_param_new = yaml_param.copy()
        yaml_param_new['num_channelsG1'] = 2*16*yaml_param['num_channelsG1']
        yaml_param_new['num_channelsG2'] = 2*16*yaml_param['num_channelsG2']        
        yaml_param_new['kernel_shape_G1'] = [3,5,7]
        yaml_param_new['pool_shape_G1'] = [2,4]
        yaml_param_new['kernel_shape_G2'] = [3,5,7]
        yaml_param_new['pool_shape_G2'] = [2,4]       
    
        #yaml_param_new['num_channelsL'] = 2*16*yaml_param['num_channelsL']

        kernel_shape_G1_range = [3,5,7]
        yaml_param_new['kernel_shape_G1'] = kernel_shape_G1_range[yaml_param['kernel_shape_G1']]

        kernel_shape_G2_range = [3,5,7]
        yaml_param_new['kernel_shape_G2'] = kernel_shape_G2_range[yaml_param['kernel_shape_G2']]

        pool_shape_G1_range = [2,4]
        yaml_param_new['pool_shape_G1'] = kernel_shape_G1_range[yaml_param['pool_shape_G1']]        

        pool_shape_G2_range = [2,4]
        yaml_param_new['pool_shape_G2'] = kernel_shape_G2_range[yaml_param['pool_shape_G2']]        
       
        lr_range = [0.001,0.005,0.01,0.05,0.1,0.5]
        yaml_param_new['lr'] = lr_range[yaml_param['lr']]

        momentum_range = [0.5,0.7,0.9]
        yaml_param_new['momentum'] = momentum_range[yaml_param['momentum']]

        L2_WD_range = [ 0.00005,0.0005,0.005]
        yaml_param_new['L2_WD'] = L2_WD_range[yaml_param['L2_WD']]

        L1_WD_range = [ 0.00005,0.0005,0.005]
        yaml_param_new['L1_WD'] = L1_WD_range[yaml_param['L1_WD']]

        decay_factor_rage = [0.01,0.1,0.5]
        yaml_param_new['decay_factor'] = decay_factor_rage[yaml_param['decay_factor']]
        return yaml_param_new

def fit_model(param,gpu):
        import os
	sub_path = os.path.dirname(os.path.realpath(__file__))
	yaml_save_path = sub_path
	#data = '/state/partition2/brats_2013_histmatched_data'
        data = '/home/local/USHERBROOKE/havm2701/data/Data'
	firstphase1_path = sub_path+'/virtual_training_fc2_template.yaml'
	model1_output = os.path.dirname(os.path.realpath(__file__))                             
	firstphase1_empty = open(firstphase1_path).read()
        import pylearn2
        pylearn2_path = pylearn2.__path__[0]
        train  = pylearn2_path + '/scripts/train.py'
        import pdb
        pdb.set_trace()

        import collections
        od = collections.OrderedDict(sorted(param.items()))
        hyp_name = ''
        for it in od.items():
  
                hyp_name +=  str(it[0])+'_'+str(it[1])+'___'

        pdb.set_trace()
	dir_path = yaml_save_path+'/'+hyp_name
	if os.path.isdir(dir_path)==False:
		os.mkdir(dir_path)
	generated_yaml_path = dir_path+'/virtual_training_fc2.yaml'
	logfile_path = dir_path +'/training.log'
	File_firstphase1 = open(generated_yaml_path,'w')
        yaml_param = param.copy()
        yaml_param = fix_yaml_params(yaml_param)
        yaml_param['data'] = data
	firstphase1 = firstphase1_empty % yaml_param
	File_firstphase1.write(firstphase1)
	File_firstphase1.close()
        gpu_flag =  'THEANO_FLAGS=device=' +gpu + ',floatX=float32'
        
	command = gpu_flag + ' ' + 'python' + ' ' + train + ' ' + generated_yaml_path + '>' + logfile_path
	print command
	os.system(command)
       
        model_path = generated_yaml_path.replace('.yaml','_best.pkl')
        accuracy = get_accuracy(model_path)
        print accuracy
        return accuracy
