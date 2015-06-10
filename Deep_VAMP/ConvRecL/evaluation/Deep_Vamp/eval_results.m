

threshold=0.25;
 dt_path= '/home/local/USHERBROOKE/havm2701/data/results_Nms/';
 gt_path= '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Test/Posframes_Annotations/';
[gt,dt] = bbGt('loadAll', gt_path, dt_path);
for i=1:length(gt)
    gtc=gt{i}
    gtc(:,1:2)= gtc(:,1:2)-gtc(:,3:4)/2;
    gt{i} = gtc
end
[gt,dt] = bbGt('evalRes',gt,dt,threshold);    
%[xs,ys,score,ref] = bbGt( 'compRoc', gt, dt, roc, ref ); NOT ENOUGH INPUT
%ARGUMENTS
dir_list = dir(dt_path);
dir_list = dir_list(3:end);
im_input_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Test/FramesPos/';
im_output_path = '/home/local/USHERBROOKE/havm2701/data/results_bb_plots/';
if ~isdir(im_output_path)
    mkdir(im_output_path)
end

for idx = 1:length(dir_list)
    dt_path_name = [dt_path,dir_list(idx).name];
    gt_path_name = [gt_path,dir_list(idx).name];
    im_name  = strrep(dir_list(idx).name, 'txt', 'png');
    im_input_name = [im_input_path,im_name];
    im_output_name = [im_output_path,im_name];
    plot_bb(im_input_name,im_output_name,gt_path_name,dt_path_name,true)
end

im_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Test/FramesPos'
plot_bb(im_path,output_path,gt_path,dt_path,true)