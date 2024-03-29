close all;
clear all;
clear classes;
clc;

% Compile the C++ library first: CompileMexKerasCppClass

%-------------------------------------------------------------------------%

% Parameters
step        = 0.1;                       % sampling step (cm)
window      = 12;
num_classes = 3;                         % number of classes
generator = 'clothoids_PRM_montecarlo';  % path planner
map = 'test';                            % map: 'void', 'cross', 'povo', 'test', 'thor1'

% Python network
neural_model = 'network_som';
som_size = [10, 10];
units    = [4, window];
classes  = num_classes;
weights_file = 'models/cross3_model';

options.save = true;
options.plot = true;

if strcmp(weights_file, 'models/cross3_model')
    class_names = {'L', 'R', 'S'};
end
% class_names = {'L-U', 'L-D', 'L-R',
%            'D-L', 'D-R', 'D-U',
%            'R-D', 'R-L', 'R-U',
%            'U-R', 'U-L', 'U-D'};

%-------------------------------------------------------------------------%

% Folder tree
addpath(genpath('../libraries/'));
addpath(genpath('./functions/'));
addpath(genpath('./synthetic_path_generators/'));
addpath(genpath('./models/'));

colors = customColors;

%% Generate data

num_traj = 1;
randomize = false;
augmentation = false;
positions = []; %[3, 1, pi/2, 13, 5, pi/2]; %[3, 3, pi/2, 7, 7, 0.0];

[Map, Pos] = map_and_positions(map, positions);
clothoids = feval(generator, Map, Pos, num_traj, randomize);
samples = get_samples(clothoids, step, augmentation);

X = [];
traj_points = [];
l = 1;
for j = 1:length(samples.s{1})-(window+1)
    Xx = samples.x{1}(j:j+(window-1)); 
    Xy = samples.y{1}(j:j+(window-1));
    Xtheta = samples.theta{1}(j:j+(window-1));
    Xkappa = samples.dtheta{1}(j:j+(window-1));
  
    % Normalize
    Xx = (Xx - min(Xx))/(max(Xx) - min(Xx));
    Xy = (Xy - min(Xy))/(max(Xy) - min(Xy));
    
    X(l,:,:) = [Xx; Xy; Xtheta; Xkappa];
    traj_points(l,:) = [samples.x{1}(j+window/2), samples.y{1}(j+window/2)];
    l = l+1;
end
clearvars Xx Xy Xtheta Xkappa;

%figure(1);
%plot(traj_points(:,1), traj_points(:,2), '*');
%axis equal, grid on;

%% Network Python class

% Import python module
pymodule = py.importlib.import_module(neural_model);
py.importlib.reload(pymodule);

% Initialize Keras class
pynet = pymodule.Network();

% Define neural network model
model = pynet.define_model(som_size, units, classes);

% Load weights
model = pynet.load_weights(weights_file);

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

%-------------------------------------------------------------------------%

% Predict
py_pred = pynet.predict(X);

som_pred = double(py_pred{1});
conf_pred = double(py_pred{2});

% Get class indexes and confidences
[Yconf, Y] = maxk(conf_pred',1);

% Plot
%plot(X(1,1:window)+samples.x{1}(1), X(1,window+1:window+window)+samples.y{1}(1), 'color', colors{1}, 'linewidth', 3);
for i = 1:5:size(traj_points,1)
    text(traj_points(i,1)+0.2, traj_points(i,2)+0.2, [class_names{Y(1,i)}, ' ', num2str(round(Yconf(1,i)*100))], 'fontsize', 16);
    %text(traj_points(i,1)+0.2, traj_points(i,2)+0.4, [class_names{Y(2,i)}, ' ', num2str(round(conf(2,i)*100))], 'fontsize', 16);
end

%% Network with C++ class

% Create C++ class instance
cppnet = KerasCpp();

%-------------------------------------------------------------------------%

cpp_pred = NaN(size(X,1), num_classes);
for i = 1:size(X,1)
    cpp_pred(i,:) = cppnet.predict(X(i,:));
end

% Get class indexes and confidences
[YYconf, YY] = maxk(cpp_pred',1);

% Plot
plot(X(1,1:window)+samples.x{1}(1), X(1,window+1:window+window)+samples.y{1}(1), 'color', colors{1}, 'linewidth', 3);
for i = 1:5:size(traj_points,1)
    text(traj_points(i,1)+0.2, traj_points(i,2)+0.4, [class_names{YY(1,i)}, ' ', num2str(round(YYconf(1,i)*100))], 'color', 'r', 'fontsize', 16);
end
