#!/bin/bash 

cd /home/local/USHERBROOKE/havm2701/git.repos/Deep_VAMP/Deep_VAMP/ConvRecL/train_real_virtual_pn
#python /home/local/USHERBROOKE/havm2701/libs/pylearn2/pylearn2/scripts/train.py virtual_training_fc2.yaml
python /home/local/USHERBROOKE/havm2701/git.repos/Deep_VAMP/Deep_VAMP/ConvRecL/make_predictions.py virtual_training_fc2_best.pkl ~/data/Data/Deep_VAMP/INRIA/Test/FramesPos_FramesNeg/ /home/local/USHERBROOKE/havm2701/git.repos/Deep_VAMP/Deep_VAMP/ConvRecL/train_real_virtual_pn/results/
matlab -nodesktop -nosplash -r "evaluate_vamp /home/local/USHERBROOKE/havm2701/git.repos/Deep_VAMP/Deep_VAMP/ConvRecL/train_real_virtual_pn/results/txt_results /home/local/USHERBROOKE/havm2701/git.repos/Deep_VAMP/Deep_VAMP/ConvRecL/train_real_virtual_pn/results/results_bb_image/;quit;"


