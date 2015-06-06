

threshold=0.25;
GT_video_path_even = 'person_and_bike_102.txt';
BB_video_path_even = '/home/local/USHERBROOKE/havm2701/Documents/MATLAB/Deep_Vamp/New Folder'
[gt,dt] = bbGt('loadAll', GT_video_path_even, BB_video_path_even);
[gt,dt] = bbGt('evalRes',gt,dt,threshold);    
[xs,ys,score,ref] = bbGt( 'compRoc', gt, dt, roc, ref );
