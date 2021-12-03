
clc;

close all;

% Parameters

num_traj = 1000;
num_points = 100;

arena = [0, 10, 0, 10];

start_x = [1, 2];
start_y = [1, 2];
start_theta = [pi/4, pi/2];
end_x = [8, 9];
end_y = [8, 9];
end_theta = [pi/4, pi/2];

% Folder tree
addpath(genpath('./Clothoids/'));

% Data
X = NaN(num_traj, 2 ,num_points);

% Plot
% npts = 50;
% figure(1);
% hold on, grid on, axis equal;
% xlim = [arena(1), arena(2)];
% ylim = [arena(3), arena(4)];

% Generate trajectories
for i = 1:num_traj
    x1 = start_x(1) + (start_x(2)-start_x(1)).*rand(1);
    y1 = start_y(1) + (start_y(2)-start_y(1)).*rand(1);
    theta1 = start_theta(1) + (start_theta(2)-start_theta(1)).*rand(1);
    x2 = end_x(1) + (end_x(2)-end_x(1)).*rand(1);
    y2 = end_y(1) + (end_y(2)-end_y(1)).*rand(1);
    theta2 = end_theta(1) + (end_theta(2)-end_theta(1)).*rand(1);
    
    % Build clothoid
    CL = ClothoidCurve();
    CL.build_G1(x1, y1, theta1, x2, y2, theta2);
    
    % Plot
    %CL.plot(npts, 'color', 'red', 'linewidth', 1);
    
    % Get points
    [xcl, ycl] = CL.points(num_points);
    X(i,1,:) = xcl;
    X(i,2,:) = ycl;
end

figure(2);
hold on, grid on, axis equal;
xlim = [arena(1), arena(2)];
ylim = [arena(3), arena(4)];
for i = 1:num_traj
    plot(squeeze(X(i,1,:)), squeeze(X(i,2,:)), 'b');
end



%%

% Import python module
pymodule = py.importlib.import_module('prova');
py.importlib.reload(pymodule);

% Initialize network and load Keras model
pynet = pymodule.Network();

model = pynet.define_model();
model.summary();

data = pynet.prepare_data(X, 80);

X_train = double(data{1});
Y_train = double(data{2});
X_valid = double(data{3});
Y_valid = double(data{4});

pynet.train_model();
%model.weights

pred = pynet.predict();
Y_pred = double(pred);

figure(3);
hold on, grid on, axis equal;
xlim = [arena(1), arena(2)];
ylim = [arena(3), arena(4)];
for i = 1:size(X_valid,1)/10
    plot(squeeze(X_valid(i,1,:)), squeeze(X_valid(i,2,:)), 'g');
    plot(squeeze(Y_valid(i,1,:)), squeeze(Y_valid(i,2,:)), 'b');
    plot(squeeze(Y_pred(i,1,:)), squeeze(Y_pred(i,2,:)), 'r');
end
