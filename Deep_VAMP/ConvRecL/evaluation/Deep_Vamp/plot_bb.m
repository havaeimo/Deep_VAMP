function plot_bb(im_path,output_path,gt_path,dt_path,g_c)
    %im = imread(im);
    gt=load(gt_path);
    dt=load(dt_path);
    [row_g col_g] = size(gt)
    [row_d col_d] = size(dt)
    if g_c==true
        gt(:,1:2)= gt(:,1:2)-gt(:,3:4)/2;
    end
    gt = [gt ones(row_g,1)]
    h=imshow(imread(im_path));
    hold on
    %bb = [gt;dt];
    %color_array = [repmat([1,0,0],row_g,1);repmat([0,1,1],row_d,1)]

    bbApply( 'draw', gt, 'g')
    bbApply( 'draw', dt, 'r')

    savefig(output_path,'png')
    hold off 
    close(gcf)
end