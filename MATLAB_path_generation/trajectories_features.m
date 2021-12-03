close all;
clear classes;
clc;

% Folder tree
addpath(genpath('./path_clustering/'));
addpath(genpath('./Clothoids/'));

% Import python module
pymodule = py.importlib.import_module('network');
py.importlib.reload(pymodule);

%%

num_traj = 50;
num_points = 50;

options_save = false;
options_plot = false;

% Call path clustering
clothoids_PRM_montecarlo_voidMap;

%%

epochs = 200;
batch  = 128;
learn_rate = 0.005;

% Initialize network and load Keras model
pynet = pymodule.Network(epochs, batch, learn_rate);

% Define neural network model
pynet.define_model(num_points);

% encoder = models{1}; encoder.summary();
% decoder = models{2}; decoder.summary();el
% autoencoder = models{3};

% Load dataset
data = pynet.prepare_data(samples, 80);

X_train = double(data{1});
X_valid = double(data{2});

% Train network
trained = pynet.train_model(X_train);

encoder = trained{1};
decoder = trained{2};
autoencoder = trained{3};

fit = trained{4};
fit_rmse = cellfun(@double,(cell(struct(fit.history).rmse)));
figure(10);
plot(fit_rmse);

% Prediction
pred = pynet.predict(autoencoder, X_valid);

X_pred = double(pred);

figure(3);
subplot(1,2,1);
hold on, grid on, axis equal;
for i = 1:size(X_train,1)
    plot(squeeze(X_train(i,1,:)), squeeze(X_train(i,2,:)), 'k');
end
subplot(1,2,2);
hold on, grid on, axis equal;
for i = 1:size(X_valid,1)
    plot(squeeze(X_valid(i,1,:)), squeeze(X_valid(i,2,:)), 'b');
    plot(squeeze(X_pred(i,1,:)), squeeze(X_pred(i,2,:)), 'r*');
end

