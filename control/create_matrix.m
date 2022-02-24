close all;
clear all;
clear classes;
clc;

% Compile the C++ library first: CompileMexKerasCppClass

%-------------------------------------------------------------------------%

% Parameters
num_traj    = 100;                       % number of trajectories
step        = 0.1;                       % sampling step (cm)
window      = 12;                        % number of samples
num_classes = 3;                         % number of classes
generator = 'clothoids_PRM_montecarlo';  % path planner
map = 'povo2Atrium';                     % map name

grid_size = 1.0;                         % grid map size

% network units: [5, 12]
% latent neurons: 5
% classes: 3
% weights file: 'models/cross3_ae3'

options.save = true;
options.plot = true;

class_names = {'L', 'R', 'S'};

%-------------------------------------------------------------------------%

% Folder tree
addpath(genpath('../libraries/'));
addpath(genpath('../MATLAB_path_generation/synthetic_path_generators/'));
addpath(genpath('../MATLAB_path_generation/functions/'));

colors = customColors;

%% Generate data

randomize    = true;
augmentation = false;
positions = [1, 5, 0.0, 9.8, 2, -pi/2];

[Map, Pos] = map_and_positions(map, positions);
clothoids = feval(generator, Map, Pos, num_traj, randomize);
samples = get_samples(clothoids, step, augmentation);

X = [];
traj_points = [];
l = 1;
for i = 1:num_traj
    for j = 1:length(samples.s{i})-(window+1)
        Xx = samples.x{i}(j:j+(window-1));
        Xy = samples.y{i}(j:j+(window-1));
        Xcos = cos(samples.theta{i}(j:j+(window-1)));
        Xsin = sin(samples.theta{i}(j:j+(window-1)));
        Xkappa = samples.dtheta{i}(j:j+(window-1));
        
        % Shift
        Xx = Xx - Xx(1);
        Xy = Xy - Xy(1);
        
        X(l,:,:) = [Xx; Xy; Xcos; Xsin; Xkappa];
        traj_points(l,:) = [samples.x{i}(j+window), samples.y{i}(j+window)]; %[samples.x{i}(j+window/2), samples.y{i}(j+window/2)];
        l = l+1;
    end
end
clearvars Xx Xy Xcos Xsin Xkappa;

%% Network inference

% Create C++ class instance
cppnet = KerasCpp();

%-------------------------------------------------------------------------%

conf_pred = NaN(size(X,1), num_classes);
for i = 1:size(X,1)
    conf_pred(i,:) = cppnet.predict(reshape(squeeze(X(i,:,:))',1,[]));
end

% Get class indexes and confidences
[Yconf, Y] = maxk(conf_pred',1);

% Plot
%figure(100);
%plot(X(1,1:window)+samples.x{1}(1), X(1,window+1:window+window)+samples.y{1}(1), 'color', colors{1}, 'linewidth', 3);
%for i = 1:3:size(traj_points,1)
%    text(traj_points(i,1)+0.2, traj_points(i,2)+0.3, [class_names{Y(1,i)}, ' ', num2str(round(Yconf(1,i)*100))], 'color', 'r', 'fontsize', 12);
%end

%% Matrix

% Obstacles in polyshapes
n = numel(Map.obstacles);
p = arrayfun(@(k) polyshape(Map.obstacles{k}(1,:), Map.obstacles{k}(2,:)), 1:n);
WallsPoly = p(1);
for i = 2:n
    WallsPoly = xor(WallsPoly, p(i));
end

% Create grid map as polyshapes
Grid = createGrid(grid_size, WallsPoly, [Map.map_res.XLocalLimits, Map.map_res.YLocalLimits]);

% Plot obstacle polyshapes and grid map
plot(WallsPoly, 'FaceColor', [0.7,0.7,0.65], 'FaceAlpha', 1, 'EdgeColor', 'k');
plot(Grid.poly(Grid.stat~=1), 'FaceColor', 'None', 'FaceAlpha', 0.1, 'EdgeColor', [0.75,0.75,0.75]);

% Build classifier matrix
points_to_search = 1:size(traj_points,1);
for i = 1:numel(Grid.poly)
    if (Grid.stat(i) == -1) | isnan(Grid.cent(i))
        continue;
    end
    
    possible_classes = [];
    searched_points = [];
    for j = points_to_search
        if norm([Grid.cent(i,:)- traj_points(j,:)]) < Grid.size*2
            if insidePolygon(Grid.poly(i).Vertices, traj_points(j,:))
                % Save classes
                possible_classes(end+1) = Y(1,j);
                searched_points(end+1) = j;
            end
        end
    end
    if not(isempty(possible_classes))
        % Choose class with max occurences
        Grid.stat(i) = mode(possible_classes);
        % Remove searched points to speed up computation
        c = ismember(points_to_search, searched_points);
        idxs = find(c);
        points_to_search(idxs) = [];
    else
        Grid.stat(i) = -1;
    end
end

figure(101);
hold on, axis equal, box on;
plot(WallsPoly, 'FaceColor', [0.7,0.7,0.65], 'FaceAlpha', 1, 'EdgeColor', 'k');
plot(Grid.poly(Grid.stat~=1), 'FaceColor', 'None', 'FaceAlpha', 0.1, 'EdgeColor', [0.75,0.75,0.75]);
for i = 1:numel(Grid.poly)
    if Grid.stat(i) == -1
        continue;
    end
    text(Grid.cent(i,1), Grid.cent(i,2), class_names{Grid.stat(i)}, 'color', 'k', 'fontsize', 12);
end
