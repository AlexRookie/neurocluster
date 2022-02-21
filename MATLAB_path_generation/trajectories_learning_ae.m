close all;
clear all;
clear classes;
clc;

% Parameters
num_traj    = 150;                       % number of trajectories
step        = 0.1;                       % sampling step (cm)
window      = 12;
num_classes = 3;                         % number of classes
generator = 'clothoids_PRM_montecarlo';  % path planner
map = 'cross';                           % map: 'void', 'cross', 'povo', 'test', 'thor1'

%num_points  = 100;                       % MINIMUM number of points

neural_model = 'network_ae_test';
epochs_unsup = 300;
epochs_sup   = 300;
batch = 64;
learn_rate = 0.001;

weights_file = 'models/cross3_ae_test';
options.save = true;
options.plot = true;

%-------------------------------------------------------------------------%

% Folder tree
addpath(genpath('../libraries/'));
addpath(genpath('./data/'));
addpath(genpath('./functions/'));
addpath(genpath('./synthetic_path_generators/'));
addpath(genpath('./models/'));

colors = customColors;

%% Data

% Load dataset
load('data/cross3b_data.mat');

% Generate dataset

%{
if strcmp(map, 'void') && (num_classes == 3)
    options.randomize = true;
    positions = [6, 10, 0.0, 12, 16,  pi/2;
                 6, 10, 0.0, 12,  4, -pi/2;
                 6, 10, 0.0, 16, 10,   0.0];
    classes = [0, 1, 2];
elseif strcmp(map, 'cross') && (num_classes == 12)
    options.randomize = true;
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
elseif strcmp(map, 'cross') && (num_classes == 3)
    options.randomize = true;
    options.augmentation = true;
    positions = [6, 10, 0.0, 10, 14, pi/2; ...
                 %6, 10, 0.0, 10, 12, pi/4;
                 6, 10, 0.0, 10, 6,  -pi/2;
                 %6, 10, 0.0, 10, 8,  -pi/4; ...
                 7, 10, 0.0, 13, 10, 0.0];
    classes = [0, 1, 2];
end

Data = cell(1,num_classes);
X = [];
y = [];
l = 1;
for i = 1:num_classes
    % Call path generator OLD
    %myTrajectories = call_generator_manual(generator, map, num_traj, step, 1);
    %trajectories = call_generator(generator, map, positions(i,:), num_traj, step, 1);
    
    % Call path generator
    [Map, Pos] = map_and_positions(map, positions(i,:), i);
    clothoids = feval(generator, Map, Pos, num_traj, options.randomize);
    samples = get_samples(clothoids, step, options.augmentation);
    Data{i} = samples;

    % Load pre-saved samples
    %samples = Data{i};

    %figure(101);
    %hold on, grid on, box on, axis equal;
    %xlabel('x (m)');
    %xlabel('y (m)');
    %title('Dataset');
    %cellfun(@plot, myTrajectories.x, myTrajectories.y);
    
    %fig2 = trajectory_features_plot(myTrajectories);
    
    % Extract samples
    for k = 1:max(size(samples.s))
        for j = 1:length(samples.s{k})-(window+1)
            Xx = samples.x{k}(j:j+(window-1));
            Xy = samples.y{k}(j:j+(window-1));
            Xcos = cos(samples.theta{k}(j:j+(window-1)));
            Xsin = sin(samples.theta{k}(j:j+(window-1)));
            Xkappa = samples.dtheta{k}(j:j+(window-1));

            X(l,:,:) = [Xx; Xy; Xcos; Xsin; Xkappa];
            y(l,1) = classes(i);
            l = l+1;
        end
    end
    %if min(cellfun(@length, myTrajectories.x)) < num_points
    %    error("Not enough points for the trajectories.");
    %end
end
save('data/cross3a_data.mat', 'X','y','Data','map','generator','positions','classes','options');
%}

% Plot dataset
% figure(1);
% hold on, grid on, box on, axis equal;
% xlabel('x (m)');
% xlabel('y (m)');
% title('Dataset');
% for i = 1:max(size(X,1))
%     plot(squeeze(X(i,1,:)), squeeze(X(i,2,:)));
% end

% Shift trajectories to origin
% shift_samples = samples - samples(:,:,1);
% Normalise samples
% norm_samples = (samples-min(samples,[],3))./(max(samples,[],3)-min(samples,[],3));
% Denormalise samples
% denorm_samples = norm_samples.*(max(samples,[],3)-min(samples,[],3)) + min(samples,[],3);

%% Network

% Import python module
pymodule = py.importlib.import_module(neural_model);
py.importlib.reload(pymodule);

% Initialize Keras class
pynet = pymodule.Network();

% Define neural network model
units = [5, 12]; %size(X,2:3);
latent_neurons = 5;
models = pynet.define_model(units, latent_neurons, num_classes);

% Load dataset
samples = pynet.prepare_data(X, y, 80, batch, true);
x_train = double(samples{1});
x_valid = double(samples{2});
y_train = double(samples{3});
y_valid = double(samples{4});

%% Train

% x1_train = x_train(:,1:2,:);
% x2_train = x_train(:,3:4,:);
% x3_train = x_train(:,5,:);

% Train unsup
trained = pynet.train_model_unsup(x_train, [], epochs_unsup, learn_rate);

encoder = trained{1};
autoencoder = trained{3};
fit_unsup = trained{5};

trained = pynet.train_model_sup(x_train, y_train, epochs_sup, learn_rate);

encoder = trained{1};
classifier = trained{4};
fit_sup = trained{5};
fit_over = trained{6};

figure(202);
subplot(1,3,1);
hold on, grid on, box on;
plot(cellfun(@double,(cell(struct(fit_unsup.history).rmse))), 'linewidth', 2);
plot(cellfun(@double,(cell(struct(fit_unsup.history).loss))), 'linewidth', 2);
legend({'rmse','loss'});
title('Unsupervised');
subplot(1,3,2);
hold on, grid on, box on;
yyaxis left;
plot(cellfun(@double,(cell(struct(fit_sup.history).accuracy))), 'linewidth', 2);
yyaxis right;
plot(cellfun(@double,(cell(struct(fit_sup.history).loss))), 'linewidth', 2);
legend({'accuracy','loss'});
title('Supervised 1');
subplot(1,3,3);
hold on, grid on, box on;
yyaxis left;
plot(cellfun(@double,(cell(struct(fit_over.history).accuracy))), 'linewidth', 2);
yyaxis right;
plot(cellfun(@double,(cell(struct(fit_over.history).loss))), 'linewidth', 2);
legend({'accuracy','loss'});
title('Supervised 2');

%% Inference

% Load weights
%trained = pynet.load_weights(weights_file);

% Predict autoencoder
pred = pynet.predict(autoencoder, x_valid);

x_pred = double(pred);

% Statistics
err_x = squeeze(mean(abs(x_valid-x_pred)))';
rmse_x = squeeze(sqrt(mean((x_valid-x_pred).^2, 1)))';

figure(10);
subplot(1,2,1);
hold on, grid on, box on;
plot(err_x, 'linewidth', 2);
xlabel('samples');
legend({'x','y','cos-theta','sin-theta','kappa'});
title('Absolute error');
subplot(1,2,2);
hold on, grid on, box on;
plot(rmse_x, 'linewidth', 2);
xlabel('samples');
legend({'x','y','cos-theta','sin-theta','kappa'});
title('Rmse');

% Predict classifier
pred = pynet.predict(classifier, x_valid);

y_pred = double(pred);

% Plot confusion matrix
[~, Y_valid] = max(y_valid');
[~, Y_pred] = max(y_pred');
confusion_matrix(Y_valid, Y_pred);

%{
figure(30);
subplot(1,3,1);
hold on, grid on, box on, axis equal;
xlabel('x');
xlabel('y');
title('Train data');
for i = 1:size(x_train,1)
    plot(squeeze(x_train(i,1,:)), squeeze(x_train(i,2,:)));
end
subplot(1,3,2);
hold on, grid on, box on, axis equal;
xlabel('x');
xlabel('y');
title('Valid data');
for i = 1:size(x_valid,1)
    plot(squeeze(x_valid(i,1,:)), squeeze(x_valid(i,2,:)));
end
subplot(1,3,3);
hold on, grid on, box on, axis equal;
xlabel('x');
xlabel('y');
title('Pred');
for i = 1:size(x_valid,1)
    plot(squeeze(x_pred(i,1,:)), squeeze(x_pred(i,2,:)));
end
%}

%% Features

[~,tmptrain] = max(y_train');
[~,tmpvalid] = max(y_valid');

class1 = [x_train(tmptrain==1,:,:); x_valid(tmpvalid==1,:,:)];
class2 = [x_train(tmptrain==2,:,:); x_valid(tmpvalid==2,:,:)];
class3 = [x_train(tmptrain==3,:,:); x_valid(tmpvalid==3,:,:)];

f1 = double(pynet.predict(encoder,class1));
f2 = double(pynet.predict(encoder,class2));
f3 = double(pynet.predict(encoder,class3));

f1a = double(pynet.predict(encoder,class1(4312,:,:)));
f1b = double(pynet.predict(encoder,class1(1,:,:)));
f2a = double(pynet.predict(encoder,class2(412,:,:)));
f2b = double(pynet.predict(encoder,class2(3212,:,:)));
f3a = double(pynet.predict(encoder,class3(1212,:,:)));
f3b = double(pynet.predict(encoder,class3(1002,:,:)));

figure(320);
bar([mean(f1)', mean(f2)', mean(f3)']);
figure(319);
subplot(1,3,1);
boxplot(f1);
subplot(1,3,2);
boxplot(f2);
subplot(1,3,3);
boxplot(f3);

figure(321);
subplot(2,3,1);
bar([f1a', f1b']);
subplot(2,3,4);
bar([f1a', f2b']);
subplot(2,3,2);
bar([f2a', f2b']);
subplot(2,3,5);
bar([f2a', f3b']);
subplot(2,3,3);
bar([f3a', f3b']);
subplot(2,3,6);
bar([f3a', f1b']);

%% Save

%pynet.save(weights_file);
