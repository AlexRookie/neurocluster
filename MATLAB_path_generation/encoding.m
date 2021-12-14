close all;
clear classes;
clc;

% Folder tree
addpath(genpath('./synthetic_path_generators/'));
addpath(genpath('./Clothoids/'));

% Parameters
num_traj   = 10;
num_points = 50;
generator = 'clothoids_PRM_montecarlo_map';
map = 'square'; % 'voidMap', 'cross'

options.save = false;
options.plot = true;

%%
% Generate data

% Call path clustering
samples = call_generators(generator, map, num_traj, num_points, options);

% Plot dataset
figure(1);
hold on, grid on, box on, axis equal;
xlabel('x (m)');
xlabel('y (m)');
title('Dataset');
plot(samples.x', samples.y');
%for i =1:num_traj
%    plot(squeeze(samples(i,1,:)), squeeze(samples(i,2,:)));
%end

%%
% Process data

figure(2);
tiledlayout(3,2, 'Padding', 'none', 'TileSpacing', 'compact');
nexttile;
plot(1:num_points, samples.theta * 180/pi);
ylim([-181,181]);
grid on;
title('Theta');
nexttile;
plot(1:num_points, samples.kappa);
grid on;
title('Curvature');
nexttile;
plot(1:num_points, samples.dtheta);
grid on;
title('∂Theta');
nexttile;
plot(1:num_points, samples.dkappa);
grid on;
title('∂Curvature');
nexttile;
plot(1:num_points, samples.dx);
ylim([-1.1,1.1]);
grid on;
title('∂x');
nexttile;
plot(1:num_points, samples.dy);
ylim([-1.1,1.1]);
grid on;
title('∂y');

