% Create video

% Alessandro Antonucci @AlexRookie
% University of Trento

clc;
clear all;
close all;

torque_constant = 0.0369;
gear_ratio = 24;

files = [7]; %[3,4,5,7]:[good, bad, so-so, good] behaviours for video

map = 'povo2Atrium';

saveVideo   = true;
plotQRcodes = false;

%-------------------------------------------------------------------------%

% Folder tree
addpath(genpath('./log_files/'));
addpath(genpath('./videos.nosync/'));
addpath(genpath('./functions/'));
addpath(genpath('../MATLAB_path_generation/synthetic_path_generators/'));

colors = customColors;
linestyles = {'-', '--', '-.', ':', '-', '--'};

past_samples = 20;

% Target points for the homography
if files <= 5
    points_top = [0    -2    -0.4  1.2  1.2  1.2  1.2  2.8  2.8  2.8;
                  5.6   0.8   1.6  0.8  2.4  4.0  5.6  1.6  3.2  4.8];  
else
    points_top = [4.6    5.2  6.0  4.4  4.4  2.8  2.8  1.2  1.2  -0.4;
                  -0.1  -1.6  0.0  0.8  2.4  1.6  3.2  0.8  2.4   1.6];
end

%-------------------------------------------------------------------------%

for i = files
    
    % Load data
    dir_files = dir('log_files/');
    dir_files = dir_files(~ismember({dir_files.name},{'.','..'}));
    load([dir_files(i).folder,'/',dir_files(i).name], '-mat', 'out', 'Grid', 'class_names');
    
    % Time vector
    t = out.tout;
    % Odometry
    theta = quaternion2theta(out.odometry.Pose.Pose.Orientation);
    pose   = [out.odometry.Pose.Pose.Position.X.Data, out.odometry.Pose.Pose.Position.Y.Data, theta];
    wheels = out.wheels_angles.angle.Data;
    % Network
    nn_in_x     = out.network.NN_inputs.out_x.Data;
    nn_in_y     = out.network.NN_inputs.out_y.Data;
    nn_in_theta = out.network.NN_inputs.out_theta.Data;
    nn_in_kappa = out.network.NN_inputs.out_kappa.Data;
    nn_in_valid = out.network.NN_inputs.out_valid.Data;
    nn_conf      = out.network.confidences.Data;
    nn_s_conf    = out.network.boosted_straight_confdence.Data;
    BOH          = out.network.confidence_star.Data;
    [~,nn_class] = max(nn_conf,[],2);
    actual_class = out.network.actual_traj_class.Data;
    % Control
    omega_control = out.omega_control.omega_control.Data;
    theta_control = out.theta_control.theta_control.Data;
    
    dt = mean(diff(t));
    
    if i == 3
        start = 201;
        stop = 801;
    elseif i == 4
        start = 1;
        stop = 621;
    elseif i == 5
        start = 1;
        stop = 701;
    elseif i == 7
        start = 241;
        stop = 660;
    end
    
    %---------------------------------------------------------------------%
    
    if saveVideo == true
        save_video = VideoWriter(['./figures/video_map', num2str(i), '.mp4'], 'MPEG-4');
        save_video.FrameRate = 1/dt;
        open(save_video);
    end
    
    fig1 = figure(1);
    dim = get(0, 'Screensize');
    set(gcf, 'Position', [0, 0, dim(3)*0.75, dim(4)*0.75]);
    set(gcf, 'Color', [1 1 1]);
    hold on, box on, axis equal, grid on;
    if i <= 5
        ax_lims = [-5, 5, -1, 7.5]; % [-3, 8, -4, 7];
    else
        ax_lims = [-5, 8, -5, 4]; % [-3, 8, -4, 7];
    end
    axis(ax_lims);
    
    xticks(ax_lims(1):2:ax_lims(2));
    yticks(ax_lims(3):2:1:ax_lims(4));
    xtickangle(0);
    set(findall(gcf,'-property','FontSize'), 'FontSize', 28);
    xlabel('x (m)', 'interpreter', 'latex', 'fontsize', 28);
    ylabel('y (m)', 'interpreter', 'latex', 'fontsize', 28);
    
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    set(gca, 'units', 'normalized'); % just making sure it's normalized
    Tight = get(gca, 'TightInset');
    set(gcf, 'InnerPosition', [0 0 (1-Tight(1)-Tight(3))*0.98*pos(3) (1-Tight(2)-Tight(4))*pos(4)]);
    %Tight = get(gca, 'TightInset');  % gives you the bording spacing between plot box and any axis labels
    %NewPos = [Tight(1) Tight(2)+0.07 1-Tight(1)-Tight(3) 1-Tight(2)-(Tight(4)+0.08)]; % new plot position [X Y W H]
    %set(gca, 'Position', NewPos);
    
    [h, trasl_x, trasl_y] = plot_map(map, false);
    %plot(Grid.poly(Grid.stat~=1), 'FaceColor', 'None', 'FaceAlpha', 0.1, 'EdgeColor', [0.75,0.75,0.75]);
    for k = 1:numel(Grid.poly)
        if Grid.stat(k) == 0 || ~(ax_lims(1)+trasl_x < mod(k,size(Grid.stat,1)) && ax_lims(2)+trasl_x > mod(k,size(Grid.stat,1))) || ~(ax_lims(3)+trasl_y < ceil(k/size(Grid.stat,1)) && ax_lims(4)+trasl_y > ceil(k/size(Grid.stat,1)))
            continue;
        end
        text(Grid.cent(k,1)-trasl_x, Grid.cent(k,2)-trasl_y, class_names{Grid.stat(k)}, 'color', [205,221,255]./255, 'fontsize', 40); %, 'FontWeight', 'bold');
    end
    if plotQRcodes == true
        r = readlines('listaQR.txt');
        qr = jsondecode(r{1}).value;
        hold on;
        for q = 1:length(qr)
            plot(qr(q).x, qr(q).y , '*');
            text(qr(q).x, qr(q).y, num2str(qr(q).id));
        end
    end
    
    for it = start+past_samples:stop%length(t)
        if exist('hg','var')
            delete(hg);
        end
        hg = gobjects();
        
        title(sprintf('t: %3.2f s', t(it)), 'FontSize', 24, 'Interpreter', 'latex');
        
        hg = plot_unicycle(hg, pose(it,1), pose(it,2), pose(it,3), 'r', 1.2);
        hg(end+1) = plot(pose(it-past_samples:it,1), pose(it-past_samples:it,2), 'r', 'linewidth', 2, 'linestyle', '-');
        hg(end+1) = plot(pose(start:it,1), pose(start:it,2), 'linewidth', 2, 'linestyle', '--');
        
        if saveVideo == true
            writeVideo(save_video, getframe(fig1));
        else
            drawnow;
            pause(dt);
        end
    end
    
    if saveVideo == true
        close(save_video);
    end
    
    %---------------------------------------------------------------------%
    
    if saveVideo == true
        save_video = VideoWriter(['./figures/video_control', num2str(i), '.mp4'], 'MPEG-4');
        save_video.FrameRate = 1/dt;
        open(save_video);
    end
    
    fig10 = figure(10);
    set(gcf, 'Position', [0, 0, dim(3), dim(4)]);
    set(gcf, 'Color', [1 1 1]);
    hold on, box on, grid on;
    xlim([t(start), t(stop)]);
    ylim([-2, 4]);
    
    set(findall(gcf,'-property','FontSize'), 'FontSize', 40);
    xlabel('t (s)', 'interpreter', 'latex', 'fontsize', 40);
    ylabel('$\tau$ (Nm)', 'interpreter', 'latex', 'fontsize', 40);
    
    for it = start+past_samples:stop%length(t)
        title(sprintf('t: %3.2f s', t(it)), 'FontSize', 24, 'Interpreter', 'latex');
        
        plot(t(1:it), (theta_control(1:it) + omega_control(1:it))/1000*torque_constant*gear_ratio, 'color', colors{1}, 'linewidth', 4, 'linestyle', linestyles{1});
        plot(t(1:it), (omega_control(1:it))/1000*torque_constant*gear_ratio, 'color', colors{2}, 'linewidth', 4, 'linestyle', linestyles{2});
        plot(t(1:it), (theta_control(1:it))/1000*torque_constant*gear_ratio, 'color', colors{3}, 'linewidth', 4, 'linestyle', linestyles{3});
        
        if saveVideo == true
            writeVideo(save_video, getframe(fig10));
        else
            drawnow;
            pause(dt);
        end
    end
    
    if saveVideo == true
        close(save_video);
    end
    %---------------------------------------------------------------------%
    
    if saveVideo == true
        save_video = VideoWriter(['./figures/video_conf', num2str(i), '.mp4'], 'MPEG-4');
        save_video.FrameRate = 1/dt;
        open(save_video);
    end
    
    fig11 = figure(11);
    set(gcf, 'Position', [0, 0, dim(3), dim(4)]);
    set(gcf, 'Color', [1 1 1]);
    hold on, box on, grid on;
    xlim([t(start), t(stop)]);
    ylim([-0.1, 1.1]);
    
    set(findall(gcf,'-property','FontSize'), 'FontSize', 40);
    xlabel('t (s)', 'interpreter', 'latex', 'fontsize', 40);
    ylabel('\epsilon', 'interpreter', 'tex', 'fontsize', 40);
    
    for it = start+past_samples:stop%length(t)
        title(sprintf('t: %3.2f s', t(it)), 'FontSize', 24, 'Interpreter', 'latex');
        
        plot(t(1:it), nn_conf(1:it,1), 'color', colors{4}, 'linewidth', 4, 'linestyle', linestyles{1});
        plot(t(1:it), nn_conf(1:it,2), 'color', colors{5}, 'linewidth', 4, 'linestyle', linestyles{2});
        plot(t(1:it), nn_conf(1:it,3), 'color', colors{6}, 'linewidth', 4, 'linestyle', linestyles{3});
        
        if saveVideo == true
            writeVideo(save_video, getframe(fig11));
        else
            drawnow;
            pause(dt);
        end
    end
    
    if saveVideo == true
        close(save_video);
    end
    
    %---------------------------------------------------------------------% 
    
    % Load video
    video = VideoReader(['./videos.nosync/', dir_files(i).name(1:end-4), '.mov']);
    load(['./videos.nosync/homography_', dir_files(i).name], '-mat', 'camera_matrix');
    
    %{
    % Show frame
    fig2 = figure(2);
    image = read(video,1);
    imshow(image);
    title('Select the points and press Enter', 'FontSize', 16);
    hold on;
    % Select points
    [px, py] = getpts;
    pixels = [px, py];
    % Highlight chosen pixels
    clf;
    imshow(image);
    title('Chosen points', 'FontSize', 16);
    hold on;
    plot(px, py, 'redx', 'MarkerSize', 10);
    for p = 1:length(pixels(:,1))
        text(pixels(p,1), pixels(p,2), (['  (' num2str(p) ')']));
    end
    hold off;
    
    camera_matrix = homography2d_solve(points_top, pixels);
    
    % Save or discard new points
    prompt = 'Save points and homography? Y/N [N]: ';
    str = input(prompt,'s');
    
    if isempty(str) % no feedback
        str = 'N';
    end
    if (str == 'Y') || (str == 'y') % yes
        save(['videos.nosync/homography_', dir_files(i).name], 'pixels', 'points_top', 'camera_matrix');
        disp('Points saved!');
    else % no
        disp('Nothing was done.');
    end
    %}
    
    % Rototranslate path
    pose_on_video = homography2d_project(camera_matrix, pose(:,1:2)');
    
    if saveVideo == true
        save_video = VideoWriter(['./figures/video', num2str(i), '.mp4'], 'MPEG-4');
        save_video.FrameRate = video.FrameRate/6;
        open(save_video);
    end
    
    t_start = t(start);
    t_on_video = (t_start + (1:video.NumFrames)*(1/video.FrameRate))';
    
    fig3 = figure(3);
    set(gcf,'color','w');
    hold on;
    
    for it = start+past_samples:2:stop
        t_now = t(it);
        frame = find(abs(t_on_video-t_now)<1e-2,1);
        
        if isempty(frame)
            continue;
        end
        
        image = read(video,frame);
        imshow(image);
        hold on;
        plot(pose_on_video(start:it,1), pose_on_video(start:it,2), 'Color', [colors{1}, 0.8], 'LineWidth', 4);
        
        if saveVideo == true
            writeVideo(save_video, getframe(fig3));
        else
            drawnow;
            pause(1/video.FrameRate);
        end
    end
    
    if saveVideo == true
        close(save_video);
    end
end
