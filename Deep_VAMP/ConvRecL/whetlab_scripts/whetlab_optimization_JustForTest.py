import generate_yaml
import pdb
access_token= '31ca9b27-81c2-42e9-a141-45047727050f'
'''
parameters = { 'learning_rate':{'min':0.001, 'max':0.1,'type':'float'},
               'init_momentum':{'min':0.1, 'max':1,'type':'float'},
               'WeightDecay':{'min':0.000001, 'max':0.01,'type':'float'},
               'L1WeightDecay':{'min':0.000001, 'max':0.01,'type':'float'},
               'decay_factor':{'min':0.001, 'max':0.1,'type':'float'},
               }
'''
outcome = {'name':'Classification accuracy'}

#param = {'learning_rate':0.1,'init_momentum':0.9, 'WeightDecay':0.0001, 'L1WeightDecay':0.001, 'decay_factor':0.01}
import whetlab
'''
scientist = whetlab.Experiment(name="extendedinput_augmented_data",
                               description="Executed in smart lab computer to optimize arc1 on augmented data.",
                               parameters=parameters,
                               outcome=outcome,
                               access_token=access_token)
'''
parameters = { 'X':{'min':0., 'max':10,'type':'float'},
               }

scientist = whetlab.Experiment(name="My Experiment",
                               #description="Executed in smart lab computer to optimize arc1 on augmented data.",
                               #parameters=parameters,
                               #outcome={'name':'accuracy'},
                               access_token=access_token)
param = scientist.suggest()
#for i in range(1):

        #param = {u'decay_factor': 0.025750000000000002, u'WeightDecay': 0.007500250000000001, u'learning_rate': 0.07525000000000001, u'L1WeightDecay': 0.007500250000000001, u'init_momentum': 0.325}
	#performance = generate_yaml.fit_model(param,0)
	#scientist.update(params, performance)
