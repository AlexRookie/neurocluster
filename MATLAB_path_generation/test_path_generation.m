close all;
clear classes;
clc;

% Folder tree
addpath(genpath('./synthetic_path_generators/'));
addpath(genpath('./Clothoids/'));

% Parameters
num_traj   = 50;
num_points = 50;
generator = 'clothoids_PRM_montecarlo_map';
map = 'cross'; % 'void', 'cross'

options.save = false;
options.plot = true;

% Generate data

% Call path clustering
samples = call_generators(generator, map, num_traj, num_points, options);