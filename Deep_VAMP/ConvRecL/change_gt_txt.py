from make_predictions_addstringtoGT import resize_ground_truth

im_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Test/FramesPos_FramesNeg' 
txt_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Test/Annotations_clean'
resize_ground_truth(im_path,txt_path,im_size=(240,320))
