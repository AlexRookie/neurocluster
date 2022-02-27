% Analise data

% Alessandro Antonucci @AlexRookie
% University of Trento

close all;
clear all;
clc;

file = 1;
map = 'povo2Atrium';
class_names = {'L', 'R', 'S'};

%-------------------------------------------------------------------------%

% Folder tree
addpath(genpath('./data/'));
addpath(genpath('./functions/'));
addpath(genpath('../MATLAB_path_generation/synthetic_path_generators/'));

files = dir('data/log_files');
files = files(~ismember({files.name},{'.','..'}));

%% Get data

% Load data
load([files(1).folder,'/',files(1).name]);

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

%% Plot

figure(1);
dim = get(0, 'Screensize');
set(gcf, 'Position', [0, 0, dim(3)*0.75, dim(4)*0.75]);
set(gcf, 'Color', [1 1 1]);
hold on, box on, axis equal, grid on;
axis([-5, 10, -5, 8]);

h = plot_map(map, false);
plot(pose(:,1), pose(:,2), 'linewidth', 2);
%quiver(pose(:,1), pose(:,2), cos(pose(:,3)), sin(pose(:,3)));

set(findall(gcf,'-property','FontSize'), 'FontSize', 22);
xlabel('x (m)', 'interpreter', 'latex', 'fontsize', 28);
ylabel('y (m)', 'interpreter', 'latex', 'fontsize', 28);

%-------------------------------------------------------------------------%

figure(2);
dim = get(0, 'Screensize');
set(gcf, 'Position', [0, 0, dim(3)*0.75, dim(4)*0.75]);
set(gcf, 'Color', [1 1 1]);
hold on, box on, grid on;

plot(theta_control, 'linewidth', 2);
plot(omega_control, 'linewidth', 2);
legend({'theta','omega'});

set(findall(gcf,'-property','FontSize'), 'FontSize', 22);
xlabel('t (s)', 'interpreter', 'latex', 'fontsize', 28);
ylabel('% (m)', 'interpreter', 'latex', 'fontsize', 28);

%% Export PDF

%{
h = gcf;
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(h,'filename','-dpdf','-r0');
%}