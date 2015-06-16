function plot_bb(im_path,output_path,gt_path,dt_path,g_c)
    %im = imread(im);
    h=figure;
    set(h,'visible','off');
    h=imshow(imread(im_path));
    hold on
    
    gt=load(gt_path);
    dt=load(dt_path);
    [row_g col_g] = size(gt);
    [row_d col_d] = size(dt);
    
    p_dt = [];
    if row_d~=0
        for idx=1:row_d
            if dt(idx,end) > 0.8
                p_dt = [p_dt; dt(idx,:)];
            end
        end

        dt = p_dt;  
    end
    if size(dt,1)~=0
       bbApply( 'draw', dt, 'r');     
    end
    
    if size(gt,1)~=0
        if g_c==true
            gt(:,1:2)= gt(:,1:2)-gt(:,3:4)/2;
        end
        gt = [gt ones(row_g,1)];

        %bb = [gt;dt];
        %color_array = [repmat([1,0,0],row_g,1);repmat([0,1,1],row_d,1)]

        bbApply( 'draw', gt, 'g');
    end


    saveas(h,output_path,'png');
    hold off ;
    close(gcf);
end