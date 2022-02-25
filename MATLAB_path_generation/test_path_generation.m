close all;
clear all;
clc;

% Folder tree
addpath(genpath('../libraries/'));
addpath(genpath('./synthetic_path_generators/'));

% Parameters
num_traj   = 1;
generator = 'clothoids_PRM_montecarlo';
map = 'test';

% Call path generator
positions = []; %[1, 6, 0.0, 10.8, 1, -pi/2];
[Map, Pos] = map_and_positions(map, positions);
randomize = false;
clothoids = feval(generator, Map, Pos, num_traj, randomize);

% Create samples
%augmentation = false;
%samples = get_samples(clothoids, step, augmentation);

% Save map
% %fh = gcf;            % access the figure handle
% %ah = fh.Children(1); % access the corresponding axes handle
% %ah.Title.String = '';
% h = gcf;
% set(h,'Units','Inches');
% pos = get(h,'Position');
% set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
% print(h,['synthetic_path_generators/maps/',map],'-dpdf','-r0');
