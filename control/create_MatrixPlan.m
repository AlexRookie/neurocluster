% Create the Matrix plan for the control

% Alessandro Antonucci @AlexRookie
% University of Trento

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

grid_points = 1.0;                       % grid map sizes
grid_thetas = 2.0;

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

%% Generate data

randomize    = true;
augmentation = false;
positions = [1, 6, 0.0, 10.8, 1, -pi/2];

[Map, Pos] = map_and_positions(map, positions);
clothoids = feval(generator, Map, Pos, num_traj, randomize);
samples = get_samples(clothoids, step, augmentation);

X = [];
traj_points = [];
traj_thetas = [];
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
        traj_thetas(l,:) = samples.theta{i}(j+window);
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
%plot(X(1,1:window)+samples.x{1}(1), X(1,window+1:window+window)+samples.y{1}(1), 'color', 'r', 'linewidth', 3);
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
clearvars p n;

% Create grid map as polyshapes
Grid  = createGrid(grid_points, WallsPoly, [Map.map_res.XLocalLimits, Map.map_res.YLocalLimits]);
Grid2 = createGrid(grid_thetas, WallsPoly, [Map.map_res.XLocalLimits, Map.map_res.YLocalLimits]);

% Build classifier matrix
points_to_search = 1:size(traj_points,1);
for i = 1:numel(Grid.poly)
    if (Grid.stat(i) == -1) | isnan(Grid.cent(i))
        Grid.stat(i) = 0;
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
        Grid.stat(i) = 0;
    end
end
clearvars c idxs possible_classes points_to_search searched_points;

% Build orientations matrix
points_to_search = 1:size(traj_points,1);
for i = 1:numel(Grid2.poly)
    if (Grid2.stat(i) == -1) | isnan(Grid2.cent(i))
        Grid2.stat(i) = 100;
        continue;
    end
    
    possible_theta = [];
    searched_points = [];
    for j = points_to_search
        if norm([Grid2.cent(i,:)- traj_points(j,:)]) < Grid2.size*2
            if insidePolygon(Grid2.poly(i).Vertices, traj_points(j,:))
                % Save theta
                possible_theta(end+1) = traj_thetas(j,1);
                searched_points(end+1) = j;
            end
        end
    end
    if not(isempty(possible_theta))
        % Get mean orientation
        Grid2.stat(i) = nanmean(possible_theta);
        % Remove searched points to speed up computation
        c = ismember(points_to_search, searched_points);
        idxs = find(c);
        points_to_search(idxs) = [];
    else
        Grid2.stat(i) = 100;
    end
end
clearvars c idxs possible_theta points_to_search searched_points;

%plot(WallsPoly, 'FaceColor', [0.7,0.7,0.65], 'FaceAlpha', 1, 'EdgeColor', 'k');
plot(Grid.poly(Grid.stat~=1), 'FaceColor', 'None', 'FaceAlpha', 0.1, 'EdgeColor', [0.75,0.75,0.75]);
for i = 1:numel(Grid.poly)
    if Grid.stat(i) == 0
        continue;
    end
    text(Grid.cent(i,1), Grid.cent(i,2), class_names{Grid.stat(i)}, 'color', 'g', 'fontsize', 18, 'FontWeight', 'bold');
end

figure(51);
hold on, box on, axis equal;
plot(WallsPoly, 'FaceColor', [0.7,0.7,0.65], 'FaceAlpha', 1, 'EdgeColor', 'k');
plot(Grid2.poly(Grid2.stat~=1), 'FaceColor', 'None', 'FaceAlpha', 0.1, 'EdgeColor', [0.75,0.75,0.75]);
for i = 1:numel(Grid2.poly)
    if abs(Grid2.stat(i)-100)<1e-3
        continue;
    end
    quiver(Grid2.cent(i,1), Grid2.cent(i,2), 0.5*cos(Grid2.stat(i)), 0.5*sin(Grid2.stat(i)), 'color', 'r', 'linewidth', 2);
end

MatrixPlan = Grid.stat;
ThetaPlan  = Grid2.stat;

save('data/atrio1.mat', 'MatrixPlan','ThetaPlan','conf_pred','Map','positions','clothoids','samples');

clearvars i j l ans;
