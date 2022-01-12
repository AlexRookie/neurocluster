close all;
clear all;
clc;

% Folder tree
addpath(genpath('./functions/'));
addpath(genpath('./synthetic_path_generators/'));
addpath(genpath('./Clothoids/'));
addpath(genpath('./C2xyz_v2/'));

% Parameters
num_traj   = 10;
num_points = 500;
generator = 'clothoids_PRM_montecarlo_map';
map = 'cross'; % 'void', 'cross'

options.save = false;
options.plot = true;

colors = customColors;

%% Generate data

% Call path clustering
samples = call_generators(generator, map, num_traj, num_points, options);

% Plot dataset
figure(1);
hold on, grid on, box on, axis equal;
xlabel('x (m)');
xlabel('y (m)');
title('Dataset');
plot(samples.x', samples.y');
%for i =1:num_traj
%    plot(squeeze(samples(i,1,:)), squeeze(samples(i,2,:)));
%end

%% Process data

% Get area limits
area_file = [map, '_areas'];
if not(isempty(area_file))
    % Open file
    fid = fopen([area_file,'.txt'],'r');
    if fid == -1
        error(['Impossible to open file: ', map_name]);
    end
    
    limits = cell(0);    
    while ~feof(fid)
        line = fgets(fid);
        pts_ini = sscanf(line, '%f %f');
        line = fgets(fid);
        pts_fin = sscanf(line, '%f %f');
        % Segment vertices
        limits(end+1,1) = {[pts_ini(1), pts_fin(1); pts_ini(2), pts_fin(2)]};
    end
    
    % Close file
    fclose(fid);
end

% Find area limits crossing
limits_on_traj = NaN(numel(limits),num_traj);
for k = 1:numel(limits)
    for i = 1:num_traj
        min_d = Inf;
        min_j = NaN;
        for j = 1:num_points
            a = limits{k}(:,1)' - limits{k}(:,2)';
            b = [samples.x(i,j), samples.y(i,j)] - limits{k}(:,2)';
            a(3) = 0;
            b(3) = 0;
            d = norm(cross(a,b)) / norm(a);
            if d < min_d && d < 0.3
                min_d = d;
                min_j = j;
            end
        end
        limits_on_traj(k,i) = min_j;
    end
end

% Plot

figure(100);
for k = 1:numel(limits)
    plot(limits{k}(1,:), limits{k}(2,:), '--', 'linewidth', 1);
    for i = 1:num_traj
        if isnan(limits_on_traj(k,i))
            continue;
        end
        plot(samples.x(i,limits_on_traj(k,i)), samples.y(i,limits_on_traj(k,i)), '.', 'color', colors{k}, 'markersize', 26, 'linestyle', 'none');
    end
end

figure(2);
tiledlayout(3,2, 'Padding', 'none', 'TileSpacing', 'compact');
nexttile;
hold on;
plot(1:num_points, samples.theta' * 180/pi);
for k = 1:numel(limits)
    for i = 1:num_traj
        if isnan(limits_on_traj(k,i))
            continue;
        end
        plot(limits_on_traj(k,i), samples.theta(i,limits_on_traj(k,i)) * 180/pi, '.', 'color', colors{k}, 'markersize', 24, 'linestyle', 'none');
    end
end
ylim([-181,181]);
grid on;
title('Theta');
nexttile;
hold on;
plot(1:num_points, samples.kappa);
for k = 1:numel(limits)
    for i = 1:num_traj
        if isnan(limits_on_traj(k,i))
            continue;
        end
        plot(limits_on_traj(k,i), samples.kappa(i,limits_on_traj(k,i)), '.', 'color', colors{k}, 'markersize', 24, 'linestyle', 'none');
    end
end
grid on;
title('Curvature');
nexttile;
hold on;
plot(1:num_points, samples.dtheta);
for k = 1:numel(limits)
    for i = 1:num_traj
        if isnan(limits_on_traj(k,i))
            continue;
        end
        plot(limits_on_traj(k,i), samples.dtheta(i,limits_on_traj(k,i)), '.', 'color', colors{k}, 'markersize', 24, 'linestyle', 'none');
    end
end
grid on;
title('∂Theta');
nexttile;
hold on;
plot(1:num_points, samples.dkappa);
for k = 1:numel(limits)
    for i = 1:num_traj
        if isnan(limits_on_traj(k,i))
            continue;
        end
        plot(limits_on_traj(k,i), samples.dkappa(i,limits_on_traj(k,i)), '.', 'color', colors{k}, 'markersize', 24, 'linestyle', 'none');
    end
end
grid on;
title('∂Curvature');
nexttile;
hold on;
plot(1:num_points, samples.dx);
for k = 1:numel(limits)
    for i = 1:num_traj
        if isnan(limits_on_traj(k,i))
            continue;
        end
        plot(limits_on_traj(k,i), samples.dx(i,limits_on_traj(k,i)), '.', 'color', colors{k}, 'markersize', 24, 'linestyle', 'none');
    end
end
ylim([-1.1,1.1]);
grid on;
title('∂x');
nexttile;
hold on;
plot(1:num_points, samples.dy);
for k = 1:numel(limits)
    for i = 1:num_traj
        if isnan(limits_on_traj(k,i))
            continue;
        end
        plot(limits_on_traj(k,i), samples.dy(i,limits_on_traj(k,i)), '.', 'color', colors{k}, 'markersize', 24, 'linestyle', 'none');
    end
end
ylim([-1.1,1.1]);
grid on;
title('∂y');
