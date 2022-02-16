close all;
clear all;
clear classes;
clc;

%-------------------------------------------------------------------------%

% Parameters
step        = 0.1;                       % sampling step (cm)
window      = 12;
num_classes = 3;                         % number of classes
generator = 'clothoids_PRM_montecarlo';  % path planner
map = 'test';                            % map: 'void', 'cross', 'povo', 'test', 'thor1'

% Python network
neural_model = 'network_ae';
units    = [4, window];
classes  = num_classes;
weights_file = 'models/cross3ae';

options.save = true;
options.plot = true;

if strcmp(weights_file, 'models/cross3ae')
    class_names = {'L', 'R', 'S'};
end

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
    
    % Shift
    Xx = Xx - Xx(1);
    Xy = Xy - Xy(1);
  
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
models = pynet.define_model(units, num_classes);

% Load weights
trained = pynet.load_weights(weights_file);

encoder = trained{1}; % encoder.summary();
decoder = trained{2};
autoencoder = trained{3};
classifier = trained{4};

%-------------------------------------------------------------------------%

% Predict autoencoder
pred = pynet.predict(autoencoder, X);
x_pred = double(pred);

% Statistics
err_x = squeeze( mean(abs(X-x_pred)) )';
rmse_x = squeeze( sqrt(mean((X-x_pred).^2, 1)) )';

figure(10);
subplot(1,2,1);
hold on, grid on, box on;
plot(err_x, 'linewidth', 2);
xlabel('samples');
legend({'x','y','theta','kappa'});
title('Absolute error');
subplot(1,2,2);
hold on, grid on, box on;
plot(rmse_x, 'linewidth', 2);
xlabel('samples');
legend({'x','y','theta','kappa'});
title('Rmse');

%-------------------------------------------------------------------------%

% Predict
pred = pynet.predict(classifier, X);
conf_pred = double(pred);

% Get class indexes and confidences
[Yconf, Y] = maxk(conf_pred',2);

% Plot
figure(100);
%plot(X(1,1:window)+samples.x{1}(1), X(1,window+1:window+window)+samples.y{1}(1), 'color', colors{1}, 'linewidth', 3);
for i = 1:3:size(traj_points,1)
    text(traj_points(i,1)+0.2, traj_points(i,2)+0.2, [class_names{Y(1,i)}, ' ', num2str(round(Yconf(1,i)*100))], 'color', 'k', 'fontsize', 12);
    text(traj_points(i,1)+0.2, traj_points(i,2)+0.3, [class_names{Y(2,i)}, ' ', num2str(round(Yconf(2,i)*100))], 'color', 'r', 'fontsize', 12);
end

%% Network with C++ class
% 
% % Create C++ class instance
% cppnet = KerasCpp();
% 
% %-------------------------------------------------------------------------%
% 
% cpp_pred = NaN(size(X,1), num_classes);
% for i = 1:size(X,1)
%     cpp_pred(i,:) = cppnet.predict(X(i,:));
% end
% 
% % Get class indexes and confidences
% [YYconf, YY] = maxk(cpp_pred',1);
% 
% % Plot
% plot(X(1,1:window)+samples.x{1}(1), X(1,window+1:window+window)+samples.y{1}(1), 'color', colors{1}, 'linewidth', 3);
% for i = 1:5:size(traj_points,1)
%     text(traj_points(i,1)+0.2, traj_points(i,2)+0.4, [class_names{YY(1,i)}, ' ', num2str(round(YYconf(1,i)*100))], 'color', 'r', 'fontsize', 16);
% end
