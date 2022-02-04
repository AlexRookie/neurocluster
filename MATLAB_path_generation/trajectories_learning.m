close all;
clear all;
clear classes;
clc;

% Parameters
num_traj    = 50;                        % number of trajectories
step        = 0.08;                      % sampling step (cm)
window      = 20;
num_classes = 12;                        % number of classes
generator = 'clothoids_PRM_montecarlo';  % path planner
map = 'cross';                            % map: 'void', 'cross', 'povo', 'test', 'thor1'

%num_points  = 100;                       % MINIMUM number of points

neural_model = 'network_som';
epochs = 1000;
%epochs_sup   = 1;
batch = 64;
learn_rate = 0.0001;
som_size = [10, 10];

%dataset = 'edinburgh_10Sep';

options.save = true;
options.plot = true;

%-------------------------------------------------------------------------%

% Folder tree
addpath(genpath('./libraries/'));
addpath(genpath('./functions/'));
addpath(genpath('./synthetic_path_generators/'));
addpath(genpath('./models/'));

colors = customColors;

% Import python module
pymodule = py.importlib.import_module(neural_model);
py.importlib.reload(pymodule);

%% Generate data

% Load dataset
%myTrajectories = load_dataset(dataset, num_points, options);

% X = [], y = [];
% for i = 1:200
%     X = [X; ones(1,30)*2];
%     y = [y; 0];
% end
% for i = 1:200
%     X = [X; ones(1,30)*5];
%     y = [y; 1];
% end
% for i = 1:200
%     X = [X; ones(1,30)*9];
%     y = [y; 2];
% end

if strcmp(map, 'void')
    positions = [6, 10, 0.0, 12, 16,  pi/2;
                 6, 10, 0.0, 12,  4, -pi/2;
                 6, 10, 0.0, 16, 10,   0.0];
    classes = [0, 1, 2];
elseif strcmp(map, 'cross')
    positions = [5,  10,  0.0,  10, 15,  pi/2;
                 5,  10,  0.0,  10, 5,  -pi/2;
                 5,  10,  0.0,  17, 10,  0.0;
                 10, 5,   pi/2, 5,  10, -pi;
                 10, 5,   pi/2, 15, 10,  0.0;
                 10, 5,   pi/2, 10, 17,  pi/2;
                 15, 10, -pi,   10, 5,  -pi/2;
                 15, 10, -pi,   10, 15,  pi/2;
                 15, 10, -pi,   3,  10, -pi;
                 10, 15, -pi/2, 15, 10,  0.0;
                 10, 15, -pi/2, 5,  10, -pi;
                 10, 15, -pi/2, 10, 3,  -pi/2;
                 ];
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
end

load('cross_data.mat');

Data = cell(1,num_classes);
X = [];
y = [];
l = 1;
for i = 1:num_classes
    % Call path generator
    [Map, Pos] = map_and_positions(map, positions(i,:));
    clothoids = feval(generator, Map, Pos, num_traj, randomize);
    samples = get_samples(clothoids, step, num_traj);
    Data{i} = samples;
    
    %myTrajectories = call_generator_manual(generator, map, num_traj, step, 1);
    %trajectories = call_generator(generator, map, positions(i,:), num_traj, step, 1);
    %Data{i} = trajectories;
    
    %figure(101);
    %hold on, grid on, box on, axis equal;
    %xlabel('x (m)');
    %xlabel('y (m)');
    %title('Dataset');
    %cellfun(@plot, myTrajectories.x, myTrajectories.y);
    
    %fig2 = trajectory_features_plot(myTrajectories);
    
    % Extract samples
    for k = 1:num_traj
        for j = 1:length(trajectories.s{k})-(window+1)
            Xx = trajectories.x{k}(j:j+(window-1)) - trajectories.x{k}(j); % shift x and y
            Xy = trajectories.y{k}(j:j+(window-1)) - trajectories.y{k}(j);
            Xtheta = trajectories.theta{k}(j:j+(window-1));
            Xkappa = trajectories.dtheta{k}(j:j+(window-1));
            %[samples_x(i,j:j+(window-1)), samples_y(i,j:j+(window-1)), samples_theta(i,j:j+(window-1))];
            
            X(l,:) = [Xx, Xy, Xtheta, Xkappa];
            y(l,1) = classes(i);
            l = l+1;
        end
    end
    %if min(cellfun(@length, myTrajectories.x)) < num_points
    %    error("Not enough points for the trajectories.");
    %end
    %samples_x = [samples_x; cell2mat(cellfun(@(X) X(1:num_points), myTrajectories.x, 'UniformOutput', false)')];
    %samples_y = [samples_y; cell2mat(cellfun(@(X) X(1:num_points), myTrajectories.y, 'UniformOutput', false)')];
    %samples_theta = [samples_theta; cell2mat(cellfun(@(X) X(1:num_points), myTrajectories.theta, 'UniformOutput', false)')];
end
save('cross_data.mat', 'X','y','Data','map','generator','positions','classes');

% Plot dataset
% figure(1);
% hold on, grid on, box on, axis equal;
% xlabel('x (m)');
% xlabel('y (m)');
% title('Dataset');
% for i = 1:size(X,1)
%     plot(X(i,1:window), X(i,window+1:window+window));
% end

% Shift trajectories to origin
%shift_samples = samples - samples(:,:,1);
% Normalise samples
%norm_samples = (samples-min(samples,[],3))./(max(samples,[],3)-min(samples,[],3));
% Denormalise samples
% denorm_samples = norm_samples.*(max(samples,[],3)-min(samples,[],3)) + min(samples,[],3);

%% Network

% Initialize Keras class
pynet = pymodule.Network();

% Define neural network model
units = size(X,2);
model = pynet.define_model(som_size, units, num_classes);

% Load dataset
samples = pynet.prepare_data(X, y, 80, batch, 1);
x_train = double(samples{1});
x_valid = double(samples{2});
y_train = double(samples{3});
y_valid = double(samples{4});

%% Train network

% Train
model = pynet.train_model(x_train, y_train, epochs, learn_rate);

% Load
%model = pynet.load_weights('model_appo');

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

%% Inference

% Predict
pred = pynet.predict(x_valid);

som_pred = double(pred{1});
y_pred = double(pred{2});

% Plot confusion matrix
[~, Y_valid] = max(y_valid');
[~, Y_pred] = max(y_pred');
confusion_matrix(Y_valid, Y_pred);

%% Save

pynet.save('cross_model');


%% LVQ network

%{

% Initialize network and load Keras model
%pynet = pymodule.Network(som_size, epochs, batch, learn_rate);
pynet = pymodule.Network();

% Define neural network model
pynet.define_model(som_size);

% Load dataset
samples = pynet.prepare_data(X, y, 80, batch, 1);

x_train = double(samples{1});
y_train = double(samples{2});
x_valid = double(samples{3});
y_valid = double(samples{4});

%% Train network

% Train
model = pynet.train_model(x_train, y_train, epochs, epochs_sup, learn_rate);

%% Inference SOM

data = pynet.get_data();
weights = double(data{1});
bias = double(data{3});
clus_labels = double(data{4});

% Get label vector
cl = reshape(clus_labels,10,10)

% Plot som weights
w = reshape(weights,10,10,[]);
figure(60);
tiledlayout(2, 2, 'Padding', 'none', 'TileSpacing', 'compact');
[mesh_x,mesh_y] = meshgrid(0.5:1:10.5,0.5:1:10.5);
for i = 1:4
    nexttile;
    hold on, axis equal, box on;
    mesh_w = zeros(size(mesh_x));
    mesh_w(1:10,1:10) = w(:,:,i);
    surf(mesh_x, mesh_y, mesh_w);
    for j = 1:10
        for k = 1:10
            text(j, k, 5, sprintf("%d",cl(j,k)));
        end
    end
    view(0,90);
end

% Predict
y_pred = pynet.predict(squeeze(x_valid(1,:,:)));
y_pred = double(y_pred);

y_pred = mode(y_pred')';

% Plot confusion matrix
confusion_matrix(y_valid, y_pred);

%%

y_pred = pynet.predict(x_valid, y_valid);
y_pred = double(y_pred);

% Plot confusion matrix
confusion_matrix(y_valid, y_pred);

data = pynet.get_data();
weights = double(data{1});
labels = double(data{2});
bias = double(data{3});
clus_labels = double(data{4});

% Get label vector
reshape(labels,10,10)

% Plot som weights
figure(50);
tiledlayout(som_size(1), som_size(2), 'Padding', 'none', 'TileSpacing', 'compact');
for k = 1:som_size(1)*som_size(2)
    nexttile;
    tmp = reshape(weights(k,:), window, num_classes);
    plot(tmp(:,1), tmp(:,2));
    grid on, box on, axis equal;
    set(gca,'visible','off');
    drawnow;
end

if options.save
    save('matlab.mat', 'weights','labels','bias');
end

%net1 = pymodule.Network();
%net1.define_model(som_size);
%net1.predict(X_valid, y_valid);
%net1.load_data(weights, labels, bias);
%net1.predict(X_valid, y_valid);

%}

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

%% Autoencoder NN

%{
% Initialize network and load Keras model
%pynet = pymodule.Network(epochs, batch, learn_rate);

% encoder = models{1}; encoder.summary();
% decoder = models{2}; decoder.summary();
% autoencoder = models{3};

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
