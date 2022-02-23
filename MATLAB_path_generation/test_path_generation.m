close all;
clear all;
clc;

% Folder tree
addpath(genpath('../libraries/'));
addpath(genpath('./synthetic_path_generators/'));

% Parameters
num_traj   = 1;
generator = 'clothoids_PRM_montecarlo';
map = 'povo';

options.randomize = false;
%options.augmentation = false;

positions = [];

% Call path generator
[Map, Pos] = map_and_positions(map, positions);
%clothoids = feval(generator, Map, Pos, num_traj, options.randomize);

% Create samples
%samples = get_samples(clothoids, step, options.augmentation);

% Save map
% %fh = gcf;            % access the figure handle
% %ah = fh.Children(1); % access the corresponding axes handle
% %ah.Title.String = '';
% h = gcf;
% set(h,'Units','Inches');
% pos = get(h,'Position');
% set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
% print(h,['synthetic_path_generators/maps/',map],'-dpdf','-r0');
