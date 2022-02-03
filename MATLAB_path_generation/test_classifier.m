close all;
clear all;
clear classes;
clc;

% Parameters
step        = 0.08;                      % sampling step (cm)
window      = 20;
num_classes = 3;                         % number of classes
generator = 'clothoids_PRM_montecarlo';  % path planner
map = 'test';                            % map: 'void', 'cross', 'povo', 'test', 'thor1'

neural_model = 'network_som';
som_size = [10, 10];
units    = 80;
classes  = num_classes;
weights_file = 'model_appo';

options.save = true;
options.plot = true;

%-------------------------------------------------------------------------%

% Folder tree
addpath(genpath('./libraries/'));
addpath(genpath('./functions/'));
addpath(genpath('./synthetic_path_generators/'));

colors = customColors;

%% Import network

% Import python module
pymodule = py.importlib.import_module(neural_model);
py.importlib.reload(pymodule);

% Initialize Keras class
pynet = pymodule.Network();

% Define neural network model
model = pynet.define_model(som_size, units, classes);

% Load weights
model = pynet.load_weights(weights_file);

%% Generate data

num_traj = 1;
trajectory = call_generator_manual(generator, map, num_traj, step, options);

X = [];
traj_points = [];
l = 1;
for j = 1:length(trajectory.s{1})-(window+1)
    Xx = trajectory.x{1}(j:j+(window-1)) - trajectory.x{1}(j); % shift x and y
    Xy = trajectory.y{1}(j:j+(window-1)) - trajectory.y{1}(j);
    Xtheta = trajectory.theta{1}(j:j+(window-1));
    Xkappa = trajectory.dtheta{1}(j:j+(window-1));
  
    X(l,:) = [Xx, Xy, Xtheta, Xkappa];
    traj_points(l,:) = [trajectory.x{1}(j+window/2), trajectory.y{1}(j+window/2)];
    l = l+1;
end

%figure(1);
%plot(traj_points(:,1), traj_points(:,2), '*');
%axis equal, grid on;

%% Classify

som_weights = double(model.get_layer('SOM').get_weights{1});

% Plot
% fig = figure(2);
% tiledlayout(som_size(1), som_size(2), 'Padding', 'none', 'TileSpacing', 'compact');
% for k = 1:som_size(1)*som_size(2)
%     nexttile;
%     hold on, axis equal, grid on, box on;
%     ylim([min(min(som_weights)), max(max(som_weights))]);
%     %set(gca,'visible','off');
%     plot(som_weights(k,1:window), som_weights(k,window+1:window+window));
%     drawnow;
% end

% Predict
pred = pynet.predict(X);

som_pred = double(pred{1});
y_pred = double(pred{2});

% Get class indexes
[~, Y] = max(y_pred');

% Plot
plot(X(1,1:window)+trajectory.x{1}(1), X(1,window+1:window+window)+trajectory.y{1}(1), 'color', colors{1}, 'linewidth', 3);
text(traj_points(1:2:end,1)+0.2, traj_points(1:2:end,2)+0.2, num2str(Y(1:2:end)'), 'fontsize', 20);
