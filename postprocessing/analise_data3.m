% Analise data

% Alessandro Antonucci @AlexRookie
% University of Trento

close all;
clear all;
clc;

torque_constant = 0.0369;
gear_ratio = 24;

% files = [1,2,3,4,5,6];
% files = [7,8,9,10,11,12];
%files = [25,26,27];
files = [24];

dim = get(0, 'Screensize');

map = 'povo2Atrium';

%-------------------------------------------------------------------------%

% Folder tree
addpath(genpath('./log_files/'));
addpath(genpath('./functions/'));
addpath(genpath('../MATLAB_path_generation/synthetic_path_generators/'));

colors = customColors;
linestyles = {'-', '--', '-.', ':', '-', '--'};

% Load data
dir_files = dir('log_files/');
dir_files = dir_files(~ismember({dir_files.name},{'.','..'}));

load([dir_files(files(1)).folder,'/',dir_files(files(1)).name], '-mat', 'Grid', 'class_names');

for i = files %1:length(dir_files)
    figure(400+i);
    set(gcf, 'Position', [0, 0, dim(3)*0.8, dim(4)]);
    set(gcf, 'Color', [1 1 1]);
    hold on, box on, axis equal, grid on;
    if i==24
        ax_lims = [-5, 5, -1, 7.5];
    else
        ax_lims = [-3, 8, -4, 4];
    end
    
    axis(ax_lims);
    xticks(ax_lims(1):1:ax_lims(2))
    yticks(ax_lims(3):1:ax_lims(4))
    xtickangle(0)
    set(findall(gcf,'-property','FontSize'), 'FontSize', 60);
    xlabel('x (m)', 'interpreter', 'latex', 'fontsize', 70);
    ylabel('y (m)', 'interpreter', 'latex', 'fontsize', 70);
    
    [h, trasl_x, trasl_y] = plot_map(map, false);
    
    %plot(Grid.poly(Grid.stat~=1), 'FaceColor', 'None', 'FaceAlpha', 0.1, 'EdgeColor', [0.75,0.75,0.75]);
    for k = 1:numel(Grid.poly)
        if Grid.stat(k) == 0 || ~(ax_lims(1)+trasl_x < mod(k,size(Grid.stat,1)) && ax_lims(2)+trasl_x > mod(k,size(Grid.stat,1))) || ~(ax_lims(3)+trasl_y < ceil(k/size(Grid.stat,1)) && ax_lims(4)+trasl_y > ceil(k/size(Grid.stat,1)))
            continue;
        end
        text(Grid.cent(k,1)-trasl_x, Grid.cent(k,2)-trasl_y, class_names{Grid.stat(k)}, 'color', [205,221,255]./255, 'fontsize', 40); %, 'FontWeight', 'bold');
    end
    
    figure(400+i+100);
    set(gcf, 'Position', [0, 0, dim(3), dim(4)]);
    set(gcf, 'Color', [1 1 1]);
    hold on, box on, grid on;
    if i==24
        xlim([0, 35]);
        ylim([-2, 4]);
    else
        ylim([-2.5,1.5]);
    end
    set(findall(gcf,'-property','FontSize'), 'FontSize', 40);
    xlabel('t (s)', 'interpreter', 'latex', 'fontsize', 40);
    ylabel('$\tau$ (Nm)', 'interpreter', 'latex', 'fontsize', 40);
    
    figure(400+i+200);
    set(gcf, 'Position', [0, 0, dim(3), dim(4)]);
    set(gcf, 'Color', [1 1 1]);
    hold on, box on, grid on;
    if i==24
        xlim([0, 35]);
    end
    ylim([0,1]);
    set(findall(gcf,'-property','FontSize'), 'FontSize', 50);
    xlabel('t (s)', 'interpreter', 'latex', 'fontsize', 50);
    ylabel('\epsilon', 'interpreter', 'tex', 'fontsize', 50);
    
    % Ship broken data
    if (i==6) || (i==10)
        continue;
    end
    
    % Get data
    
    % Load data
    load([dir_files(i).folder,'/',dir_files(i).name], '-mat', 'out', 'Grid', 'grid_size', 'class_names');
    % do not load: Map, positions, position, ThetaPlan
    
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
    
    %---------------------------------------------------------------------%
    
    % Plot
    figure(400+i);
    plot(pose(:,1), pose(:,2), 'color', colors{mod(3,10)+1}, 'linewidth', 3, 'linestyle', linestyles{mod(0,6)+1});
    %quiver(pose(:,1), pose(:,2), cos(pose(:,3)), sin(pose(:,3)));
    hg = [];
    hg = plot_unicycle(hg, pose(10,1), pose(10,2), pose(10,3), 'k', 1.6);
    
    % Export PDF
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    set(gca, 'units', 'normalized'); %Just making sure it's normalized
    Tight = get(gca, 'TightInset')
    set(gcf, 'InnerPosition', [0 0 (1-Tight(1)-Tight(3))*0.98*pos(3) (1-Tight(2)-Tight(4))*pos(4)]);
    Tight = get(gca, 'TightInset')  %Gives you the bording spacing between plot box and any axis labels
                                     %[Left Bottom Right Top] spacing
    NewPos = [Tight(1) Tight(2)+0.07 1-Tight(1)-Tight(3) 1-Tight(2)-(Tight(4)+0.08)]; %New plot position [X Y W H]
    set(gca, 'Position', NewPos);
    print(gcf,strcat('figures/plotMap',num2str(i)),'-dpdf','-r0');
    saveas(gcf,strcat('figures/plotMap',num2str(i)));
    
    %---------------------------------------------------------------------%
    
    figure(400+i+100);
    %plot(t, (theta_control + omega_control)/1000*torque_constant*gear_ratio, 'color', colors{mod(0,10)+1}, 'linewidth', 3, 'linestyle', linestyles{mod(0,6)+1});
    plot(t, (theta_control + omega_control)/1000*torque_constant*gear_ratio, 'color', colors{mod(i,10)+1}, 'linewidth', 4, 'linestyle', linestyles{mod(i,6)+1});
    plot(t, (omega_control)/1000*torque_constant*gear_ratio, 'color', colors{mod(i,10)+2}, 'linewidth', 4, 'linestyle', linestyles{mod(i,6)+2});
    plot(t, (theta_control)/1000*torque_constant*gear_ratio, 'color', colors{mod(i,10)+3}, 'linewidth', 4, 'linestyle', linestyles{mod(i,6)+3});
    
    % Export PDF
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);

    set(gca, 'units', 'normalized'); %Just making sure it's normalized
    Tight = get(gca, 'TightInset');  %Gives you the bording spacing between plot box and any axis labels
                                     %[Left Bottom Right Top] spacing
    NewPos = [Tight(1) Tight(2) 1-Tight(1)-Tight(3) 1-Tight(2)-Tight(4)]; %New plot position [X Y W H]
    set(gca, 'Position', NewPos);
    
    print(gcf,strcat('figures/plotControl',num2str(i)),'-dpdf','-r0');
    saveas(gcf,strcat('figures/plotControl',num2str(i)));
    
    %---------------------------------------------------------------------%
    
    figure(400+i+200);
    %plot(t, (theta_control + omega_control)/1000*torque_constant*gear_ratio, 'color', colors{mod(0,10)+1}, 'linewidth', 3, 'linestyle', linestyles{mod(0,6)+1});
    plot(t, nn_conf(:,1), 'color', colors{mod(i,10)+4}, 'linewidth', 4, 'linestyle', linestyles{mod(i,6)+1});
    plot(t, nn_conf(:,2), 'color', colors{mod(i,10)+5}, 'linewidth', 4, 'linestyle', linestyles{mod(i,6)+2});
    plot(t, nn_conf(:,3), 'color', colors{mod(i,10)+6}, 'linewidth', 4, 'linestyle', linestyles{mod(i,6)+3});
    
    % Export PDF
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    
   % set(gca, 'units', 'normalized'); %Just making sure it's normalized
    Tight = get(gca, 'TightInset');  %Gives you the bording spacing between plot box and any axis labels
    %[Left Bottom Right Top] spacing
  %  NewPos = [Tight(1) Tight(2) 1-Tight(1)-Tight(3) 1-Tight(2)-Tight(4)]; %New plot position [X Y W H]
  %  set(gca, 'Position', NewPos);
    
    print(gcf,strcat('figures/plotConfidence',num2str(i)),'-dpdf','-r0');
    saveas(gcf,strcat('figures/plotConfidence',num2str(i)));
end