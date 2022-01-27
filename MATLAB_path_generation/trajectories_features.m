close all;
clear classes;
clc;

% Parameters
num_traj    = 50;                        % number of trajectories
num_points  = 150;                       % MINIMUM number of points
num_classes = 3;                         % number of classes
generator = 'clothoids_PRM_montecarlo';  % path planner
map = 'void';                            % map: 'void', 'cross', 'povo', 'test', 'thor1'
window = 20;

%dataset = 'edinburgh_10Sep';

options.save = false;
options.plot = true;

%-------------------------------------------------------------------------%

% Folder tree
addpath(genpath('./libraries/'));
addpath(genpath('./functions/'));
addpath(genpath('./synthetic_path_generators/'));
addpath(genpath('./datasets/'));

colors = customColors;

% Import python module
pymodule = py.importlib.import_module('network_lvq');
py.importlib.reload(pymodule);

%% Generate data

% Load dataset
%myTrajectories = load_dataset(dataset, num_points, options);

%load('data.mat');

%{
X = [];
y = [];
for i = 1:200
    X = [X; ones(1,30)*2];
    y = [y; 0];
end
for i = 1:200
    X = [X; ones(1,30)*5];
    y = [y; 1];
end
for i = 1:200
    X = [X; ones(1,30)*9];
    y = [y; 2];
end
%}

positions = [6, 10, 0.0, 12, 16,  pi/2;
             6, 10, 0.0, 12,  4, -pi/2;
             6, 10, 0.0, 16, 10,   0.0];

Xx = [];
Xy = [];
Xtheta = [];
Xkappa = [];
y = [];
encoding = [];
l = 1;

for i = 1:num_classes
    % Call path generator
    %myTrajectories = call_generator_manual(generator, map, num_traj, num_points, options);
    pos_1 = [6, 10, 0.0];
    pos_2 = [12, 16, pi/2];
    myTrajectories = call_generator(generator, map, positions(i,:), num_traj, num_points, options);
    
    %fig2 = trajectories_visualizer(myTrajectories);
    %figure(101);
    %hold on, grid on, box on, axis equal;
    %xlabel('x (m)');
    %xlabel('y (m)');
    %title('Dataset');
    %cellfun(@plot, myTrajectories.x, myTrajectories.y);
    
    % Extract samples
    for k = 1:num_traj
        for j = 1:length(myTrajectories.s{k})-(window-1)
            Xx(l,:) = myTrajectories.x{k}(j:j+(window-1));
            Xy(l,:) = myTrajectories.y{k}(j:j+(window-1));
            Xtheta(l,:) = myTrajectories.theta{k}(j:j+(window-1));
            Xkappa(l,:) = myTrajectories.dtheta{k}(j:j+(window-1)); %[samples_x(i,j:j+(window-1)), samples_y(i,j:j+(window-1)), samples_theta(i,j:j+(window-1))];
            y(l,1) = i;
            l = l+1;
        end
    end
    %if min(cellfun(@length, myTrajectories.x)) < num_points
    %    error("Not enough points for the trajectories.");
    %end
    %samples_x = [samples_x; cell2mat(cellfun(@(X) X(1:num_points), myTrajectories.x, 'UniformOutput', false)')];
    %samples_y = [samples_y; cell2mat(cellfun(@(X) X(1:num_points), myTrajectories.y, 'UniformOutput', false)')];
    %samples_theta = [samples_theta; cell2mat(cellfun(@(X) X(1:num_points), myTrajectories.theta, 'UniformOutput', false)')];
    %encoding = [encoding; ones(num_traj,1)*(i-1)];
end

% Plot dataset
figure(1);
hold on, grid on, box on, axis equal;
xlabel('x (m)');
xlabel('y (m)');
title('Dataset');
plot(samples_x', samples_y');

% Prepare data
X = [];
y = [];
k = 1;
for i = 1:size(samples_x,1)
    for j = 1:num_points-(window-1)
         X(k,:) = samples_theta(i,j:j+(window-1)); %[samples_x(i,j:j+(window-1)), samples_y(i,j:j+(window-1)), samples_theta(i,j:j+(window-1))];
         y(k,1) = encoding(i);
         k = k+1;
    end
end

% Shift trajectories to origin
%shift_samples = samples - samples(:,:,1);
% Normalise samples
%norm_samples = (samples-min(samples,[],3))./(max(samples,[],3)-min(samples,[],3));
% Denormalise samples
% denorm_samples = norm_samples.*(max(samples,[],3)-min(samples,[],3)) + min(samples,[],3);

%% Train network

epochs = 500;
batch  = 32;
learn_rate = 0.05;
som_size = [10, 10];

% Initialize network and load Keras model
%pynet = pymodule.Network(epochs, batch, learn_rate);
%pynet = pymodule.Network(som_size, epochs, batch, learn_rate);
pynet = pymodule.Network();

% Define neural network model
pynet.define_model(som_size);

% encoder = models{1}; encoder.summary();
% decoder = models{2}; decoder.summary();
% autoencoder = models{3};

% Load dataset
data = pynet.prepare_data(X, y, 80, batch, 1);

X_train = double(data{1});
y_train = double(data{2});
X_valid = double(data{3});
y_valid = double(data{4});

%%

% Train network
model = pynet.train_model(X_train, y_train', epochs, learn_rate);

y_pred = pynet.predict(X_valid, y_valid);
y_pred = double(y_pred);

%% SOM

%{
som = SOMSimple(10, epochs, samples, 0.1, 0.05, 20, 0.05, 2);

% Plot
som_weights = double(model.get_layer('SOM').get_weights{1});
fig = figure(2);
tiledlayout(som_size(1), som_size(2), 'Padding', 'none', 'TileSpacing', 'compact');
for k = 1:som_size(1)*som_size(2)
    nexttile;
    tmp = reshape(som_weights(k,:), num_points, 2);
    plot(tmp(1,:), tmp(2,:));
    grid on, box on;
    %set(gca,'visible','off');
end

%-------------------------------------------------------------------------%

% Create a Self-Organizing Map
net = selforgmap([som_size], 'topologyFcn', 'gridtop');
net.trainParam.epochs = epochs;

aaa = reshape(X_train, 164, 200);
bbb = reshape(X_valid, 41, 200);

% Train the Network
[net,tr] = train(net,aaa);

% Test the Network
y = net(bbb);

% View the Network
view(net)

op_som = vec2ind(net(aaa))';

op_som_7 = find(op_som == 7);
aaa_7 = aaa(op_som_7)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotsomtop(net)
%figure, plotsomnc(net)
%figure, plotsomnd(net)
%figure, plotsomplanes(net)
figure, plotsomhits(net,aaa(1,:,:))
figure, plotsompos(net,aaa(1,:,:))

%-------------------------------------------------------------------------%

angles = 0:0.5*pi/99:0.5*pi;
X = [sin(angles); cos(angles)];
plot(X(1,:),X(2,:),'+r')

net = selforgmap(10);

net.trainParam.epochs = 10;
net = train(net,X);

plotsompos(net)

x = [0.4;0.9];
a = net(x)

vec2ind(a)'

%-------------------------------------------------------------------------%
%}

%%

% Autoencoder NN

%{
% encoder = trained{1};
% decoder = trained{2};
% autoencoder = trained{3};

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
%}
