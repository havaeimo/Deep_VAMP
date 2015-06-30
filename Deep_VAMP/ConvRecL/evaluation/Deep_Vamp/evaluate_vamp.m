function evaluate_vamp(input_path,im_output_path)
nms_threshold = 0.3;
eval_threshold = 0.25;
%input_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/ROC_test/dt/';

%input_path = '/home/local/USHERBROOKE/havm2701/git.repos/Deep_VAMP/Deep_VAMP/ConvRecL/train_real_pn/results_new/';
dir_list = dir(input_path);
%im_output_path = '/home/local/USHERBROOKE/havm2701/ROC_Test/';

for i=1:length(dir_list)
    if strfind( dir_list(i).name,'txt')
        path1 = [input_path,dir_list(i).name];
        make_Nms_file(path1,nms_threshold)
    end
end

evaluate(input_path,im_output_path,eval_threshold,false)  

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

  function evaluate(dt_path,im_output_path,eval_threshold,plot_bb_flag)  
    gt_path ='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Test/Annotations_clean/';
    im_input_path = '/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/INRIA/Test/Frames_resized/';
    if ~isdir(im_output_path)
        mkdir(im_output_path)
    end
%Get the name and path of every file. Used when plotting the bounding boxes    
    file_paths = bbGt('getFiles', {dt_path});
%Load files
    [gt,dt] = bbGt('loadAll', gt_path, dt_path);
    for i=1:length(gt)
        gtc=gt{i};
        gtc(:,1:2)= gtc(:,1:2)-gtc(:,3:4)/2;
        gt{i} = gtc;
    end
    [gt,dt] = bbGt('evalRes',gt,dt,eval_threshold,1);    

    
    %ARGUMENTS


    
%% Plot ROC and missrate curves
    %ROC curve
    [xs_roc,ys_roc,score_roc] = bbGt( 'compRoc', gt, dt,1); %NOT ENOUGH INPUT
    h=figure;
    set(h,'visible','off');
    h=plot(xs_roc,ys_roc,'r.');
    xlabel('FPPI')
    ylabel('TP')
    saveas(h,[im_output_path,'ROC.png'],'png');

    hold off ;
    close(gcf);
    %PR curve
    [xs_pr,ys_pr,score_pr] = bbGt( 'compRoc', gt, dt,0); 
    g=figure;
    set(g,'visible','off');
    g=plot(xs_pr,ys_pr,'g.');
    xlabel('recall')
    ylabel('precision')
    saveas(g,[im_output_path,'PR.png'],'png');
    close(gcf);
%% Plot the bounding boxes
   if plot_bb_flag == true
        for idx = 1:length(file_paths)        
            [pathstr,name,ext]= fileparts(file_paths{i});
            %dt_path_name = [dt_path,name];
            %gt_path_name = [gt_path,name];
            %im_name  = strrep(dir_list(idx).name, 'txt', 'png');
            im_input_name = [im_input_path,name,'.png'];
            im_output_name = [im_output_path,name,'.png'];
            plot_bb(im_input_name,im_output_name,gt{i},dt{i})
        end
   end
  end