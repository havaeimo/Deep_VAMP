function main()
input_path = '/home/local/USHERBROOKE/havm2701/data/deep_vamp_results2/';
dir_list = dir(input_path);
result_path = '/home/local/USHERBROOKE/havm2701/data/results_Nms/';
if ~isdir(result_path)
    mkdir(result_path)
end

for i=1:length(dir_list)
    if strfind( dir_list(i).name,'txt')
        path1 = [input_path,dir_list(i).name];
        make_Nms_file(path1,dir_list(i).name,result_path)
    end
end


end


function make_Nms_file(input_path, name,result_path)
        bb_file = load(input_path);
        bbs = bbNms(bb_file);
        save_path = [result_path,name];
        fid = fopen(save_path,'wt');
        for ii = 1:size(bbs,1)
            fprintf(fid,'%g\t',bbs(ii,:));
            fprintf(fid,'\n');
        end
        fclose(fid);
end