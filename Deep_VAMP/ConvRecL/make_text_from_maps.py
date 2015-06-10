from make_predictions import generate_bb_score_file_from_maps

path = '/home/local/USHERBROOKE/havm2701/data/deep_vamp_results2'
generate_bb_score_file_from_maps(path,[0.65,0.75,1],'_',0.01)
