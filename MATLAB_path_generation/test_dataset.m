close all;
clear all;
clc;

options.save = false;
options.plot = true;

% Folder tree
addpath(genpath('./functions/'));
addpath(genpath('./Clothoids/'));
addpath(genpath('./datasets/'));

colors = customColors;

%% Load data

files = dir(fullfile('./datasets/*.mat'));

% Plot dataset
figure(1);
tiledlayout(3, 3, 'Padding', 'none', 'TileSpacing', 'compact');
for i = 1:numel(files)
    load(files(i).name);
    nexttile;
    hold on, grid on, box on, axis equal;
    axis(Data.AxisLim);
    xlabel('x (m)');
    xlabel('y (m)');
    title(files(i).name, 'interpreter', 'latex');
    cellfun(@(X) plot(X(:,1), X(:,2)), Data.Humans);
    drawnow;
end

disp('All done folks!');
