close all;
clear classes;
clc;

% Folder tree
addpath(genpath('./synthetic_path_generators/'));
addpath(genpath('./Clothoids/'));

% Parameters
num_traj   = 10;
num_points = 50;
generator = 'clothoids_PRM_montecarlo_void';
map = 'cross'; % 'voidMap', 'cross'

options.save = false;
options.plot = true;

%%
% Generate data

% Call path clustering
samples = call_generators(generator, map, num_traj, num_points, options);

% Plot dataset
figure(1);
hold on, grid on, box on, axis equal;
xlabel('x');
xlabel('y');
title('Dataset');
plot(samples.x', samples.y');
%for i =1:num_traj
%    plot(squeeze(samples(i,1,:)), squeeze(samples(i,2,:)));
%end

