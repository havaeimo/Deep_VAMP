#!/bin/bash 

cd /home/local/USHERBROOKE/havm2701/scripts/Deep_VAMP/ConvRecL
python /home/local/USHERBROOKE/havm2701/libs/pylearn2/pylearn2/scripts/train.py virtual_training_fc2.yaml
python make_predictions.py virtual_training_fc2_best.pkl
