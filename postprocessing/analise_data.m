% Analise data

% Alessandro Antonucci @AlexRookie
% University of Trento

close all;
clear all;
clc;

files = [1,2,3,4,5,6];
%files = [7,8,9,10,11,12];

map = 'povo2Atrium';

linestyles = {'-', '--', '-.', ':', '-', '--'};

%-------------------------------------------------------------------------%

% Folder tree
addpath(genpath('./data/'));
addpath(genpath('./functions/'));
addpath(genpath('../MATLAB_path_generation/synthetic_path_generators/'));

colors = customColors;

dir_files = dir('data/log_files');
dir_files = dir_files(~ismember({dir_files.name},{'.','..'}));

load([dir_files(files(1)).folder,'/',dir_files(files(1)).name], '-mat', 'Grid', 'class_names');

figure(1);
dim = get(0, 'Screensize');
set(gcf, 'Position', [0, 0, dim(3)*0.75, dim(4)*0.75]);
set(gcf, 'Color', [1 1 1]);
hold on, box on, axis equal, grid on;
axis([-5, 10, -5, 8]);
set(findall(gcf,'-property','FontSize'), 'FontSize', 22);
xlabel('x (m)', 'interpreter', 'latex', 'fontsize', 28);
ylabel('y (m)', 'interpreter', 'latex', 'fontsize', 28);

[h, trasl_x, trasl_y] = plot_map(map, false);

%plot(Grid.poly(Grid.stat~=1), 'FaceColor', 'None', 'FaceAlpha', 0.1, 'EdgeColor', [0.75,0.75,0.75]);
for k = 1:numel(Grid.poly)
    if Grid.stat(k) == 0
        continue;
    end
    text(Grid.cent(k,1)-trasl_x, Grid.cent(k,2)-trasl_y, class_names{Grid.stat(k)}, 'color', 'g', 'fontsize', 18); %, 'FontWeight', 'bold');
end

figure(2);
dim = get(0, 'Screensize');
set(gcf, 'Position', [0, 0, dim(3)*0.75, dim(4)*0.75]);
set(gcf, 'Color', [1 1 1]);
hold on, box on, grid on;
set(findall(gcf,'-property','FontSize'), 'FontSize', 22);
xlabel('t (s)', 'interpreter', 'latex', 'fontsize', 28);
ylabel('', 'interpreter', 'latex', 'fontsize', 28);

for i = files %1:length(dir_files)
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
    figure(1);
    plot(pose(:,1), pose(:,2), 'color', colors{mod(i,10)+1}, 'linewidth', 2, 'linestyle', linestyles{mod(i,6)+1});
    %quiver(pose(:,1), pose(:,2), cos(pose(:,3)), sin(pose(:,3)));
    
    figure(2);
    plot(t, theta_control + omega_control, 'color', colors{mod(i,10)+1}, 'linewidth', 2, 'linestyle', linestyles{mod(i,6)+1});
end

% Export PDF
%{
h = gcf;
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(h,'filename','-dpdf','-r0');
%}