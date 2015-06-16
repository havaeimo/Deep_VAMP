function evaluate_vamp(input_path,im_output_path)
nms_threshold = 0.3;
eval_threshold = 0.25;
%input_path = '/home/local/USHERBROOKE/havm2701/git.repos/Deep_VAMP/Deep_VAMP/ConvRecL/train_real_pn/results_new/';
dir_list = dir(input_path);



for i=1:length(dir_list)
    if strfind( dir_list(i).name,'txt')
        path1 = [input_path,dir_list(i).name];
        make_Nms_file(path1,nms_threshold)
    end
end

evaluate(input_path,im_output_path,eval_threshold)  

end


function make_Nms_file(input_path,threshold)
        bb_file = load(input_path);
        bbs = bbNms(bb_file,'overlap', threshold);
        save_path = input_path;%[result_path,name];
        fid = fopen(save_path,'wt');
        for ii = 1:size(bbs,1)
            fprintf(fid,'%g\t',bbs(ii,:));
            fprintf(fid,'\n');
        end
        fclose(fid);
end

  function evaluate(dt_path,im_output_path,eval_threshold)  
    gt_path ='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Test/Annotations_clean/';
    [gt,dt] = bbGt('loadAll', gt_path, dt_path);
    for i=1:length(gt)
        gtc=gt{i};
        gtc(:,1:2)= gtc(:,1:2)-gtc(:,3:4)/2;
        gt{i} = gtc;
    end
    [gt,dt] = bbGt('evalRes',gt,dt,eval_threshold);    
    [xs,ys,score] = bbGt( 'compRoc', gt, dt,1); %NOT ENOUGH INPUT
    %ARGUMENTS


    dir_list = dir(dt_path);
    dir_list = dir_list(3:end);
    im_input_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Test/Frames_resized/';
    %im_output_path = '/home/local/USHERBROOKE/havm2701/git.repos/Deep_VAMP/Deep_VAMP/ConvRecL/train_real_pn/bb_plot_3/';
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
  end