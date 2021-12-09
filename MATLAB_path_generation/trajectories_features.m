close all;
clear classes;
clc;

% Folder tree
addpath(genpath('./synthetic_path_generators/'));
addpath(genpath('./Clothoids/'));

% Import python module
pymodule = py.importlib.import_module('network');
py.importlib.reload(pymodule);

%%
% Generate data

num_traj = 300;
num_points = 50;

options_save = false;
options_plot = false;

% Call path clustering
clothoids_PRM_montecarlo_voidMap;

% Plot dataset
figure(1);
hold on, grid on, box on, axis equal;
xlabel('x');
xlabel('y');
title('Dataset');
for i =1:num_traj
    plot(squeeze(samples(i,1,:)), squeeze(samples(i,2,:)));
end

% Shift trajectories to origin
%shift_samples = samples - samples(:,:,1);

% Normalise samples
norm_samples = (samples-min(samples,[],3))./(max(samples,[],3)-min(samples,[],3));

% Denormalise samples
% denorm_samples = norm_samples.*(max(samples,[],3)-min(samples,[],3)) + min(samples,[],3);

%%
% Train network

epochs = 300;
batch  = 128;
learn_rate = 0.005;

% Initialize network and load Keras model
pynet = pymodule.Network(epochs, batch, learn_rate);

% Define neural network model
pynet.define_model(num_points);

% encoder = models{1}; encoder.summary();
% decoder = models{2}; decoder.summary();
% autoencoder = models{3};

% Load dataset
data = pynet.prepare_data(norm_samples, 80);

X_train = double(data{1});
X_valid = double(data{2});

% Train network
trained = pynet.train_model(X_train);

encoder = trained{1};
decoder = trained{2};
autoencoder = trained{3};

fit = trained{4};
fit_rmse = cellfun(@double,(cell(struct(fit.history).rmse)));
figure(2);
plot(fit_rmse);

% Prediction autoencoder
pred = pynet.predict(autoencoder, X_valid);

X_pred = double(pred);

% Plot
figure(3);
hold on, grid on, box on, axis equal;
xlabel('x');
xlabel('y');
title('Train data');
for i = 1:size(X_train,1)
    plot(squeeze(X_train(i,1,:)), squeeze(X_train(i,2,:)), 'k');
end
figure(4);
hold on, grid on, box on, axis equal;
xlabel('x');
xlabel('y');
title('Valid data');
for i = 1:size(X_valid,1)
    plot(squeeze(X_valid(i,1,:)), squeeze(X_valid(i,2,:)), 'b.-');
end
figure(5);
hold on, grid on, box on, axis equal;
xlabel('x');
xlabel('y');
title('Pred');
for i = 1:size(X_valid,1)
    plot(squeeze(X_pred(i,1,:)), squeeze(X_pred(i,2,:)), 'r.-');
end

% Statistics
err_x = squeeze( mean(abs( X_valid(:,1,:)-X_pred(:,1,:)) ) );
err_y = squeeze( mean(abs( X_valid(:,2,:)-X_pred(:,2,:)) ) );

rmse_x = squeeze( sqrt( mean( (X_valid(:,1,:)-X_pred(:,1,:)).^2, 1) ) );
rmse_y = squeeze( sqrt( mean( (X_valid(:,2,:)-X_pred(:,2,:)).^2, 1) ) );

figure(10);
subplot(1,2,1);
hold on, grid on, box on;
plot(err_x);
plot(err_y);
xlabel('samples');
legend({'x','y'});
title('Absolute error');
subplot(1,2,2);
hold on, grid on, box on;
plot(rmse_x);
plot(rmse_y);
xlabel('samples');
legend({'x','y'});
title('Rmse');
