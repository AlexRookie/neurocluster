% Parse Exp dataset and create .mat files
% Alessandro Antonucci @AlexRookie
% University of Trento

clc;
close all;
clear all;

%==========================================================================

options.save = false; % save results
options.plot = true; % show plot
options.show = false; % show statistics

num_of_exp = 30; % experiment IDs
date = '0120';   % experiments date

% Dataset parameters
map_angle_offset = -0.1682;

saveflag = true; % save results
plotflag = false; % plot

%==========================================================================

% Folder tree
addpath(genpath('./logs'));

% Data struct
Data.NumTraj = [];
Data.Humans = [];
Data.SamplingTime = [];
Data.Map = [];

Humans = cell(0);
frame = 1;

if plotflag == true
    figure(1);
    hold on, box on, grid on, axis equal;
    xlabel('x (m)', 'interpreter', 'latex', 'fontsize', 18);
    ylabel('y (m)', 'interpreter', 'latex', 'fontsize', 18);
    plot_obstacles('./maps/mappaRobodogC.txt', 'k');
end

% Loop over data
for exp = 1:num_of_exp
    % Load data
    file = sprintf('data%02d',exp);
    fprintf("File %d\n", exp);
    [human, loc, ~, ~, clusters, lidar, ~] = load_robotlog('./logs', file);
    
    % Synchronize times
    dt = 0.100;

    loc2 = [];
    k = 1;
    for i = 1:length(human.t)
        % Find closest timestamp (loc is faster than human, so we always have data)
        [~, idx] = min(abs(loc.t_rcv - human.t_rcv(i)));
        % New struct
        loc2.t_rcv(k,1) = loc.t_rcv(idx,1);
        loc2.pose(k,:)  = loc.pose(idx,:);
        k = k+1;
    end

    clusters2 = [];
    k = 1;
    for i = 1:length(human.t)
        % Find closest timestamp
        [dist, idx] = min(abs(clusters.t_rcv - human.t_rcv(i)));
        % New struct
        if (dist./1000 < dt)
            clusters2.t_rcv(k,1)     = clusters.t_rcv(idx,1);
            clusters2.centroids(k,1) = clusters.centroids(idx,1);
            for j = 1:size(clusters2.centroids{k,1},2)
                clusters2.points(k,j) = clusters.points(idx,j);
            end
            clusters2.id(k,1)      = clusters.id(idx,1);
            clusters2.type(k,1)    = clusters.type(idx,1);
            clusters2.visible(k,1) = clusters.visible(idx,1);
        else
            clusters2.t_rcv(k,1)     = human.t_rcv(i);
            clusters2.centroids(k,:) = {[]};
            clusters2.points(k,1)    = {[]};
            clusters2.id(k,1)        = {[]};
            clusters2.type(k,1)      = {[]};
            clusters2.visible(k,1)   = {[]};
        end
        k = k+1;
    end
    
    lidar2 = [];
    k = 1;
    for i = 1:length(human.t)
        % Find closest timestamp
        [dist, idx] = min(abs(lidar.t_rcv - human.t_rcv(i)));
        % New struct
        if (dist./1000 < dt)
            lidar2.t_rcv(k,1)  = lidar.t_rcv(idx,1);
            lidar2.size(k,1)   = lidar.size(idx,1);
            lidar2.points(k,1) = lidar.points(idx,1);
        else
            lidar2.t_rcv(k,1)  = human.t_rcv(i);
            lidar2.size(k,1)   = 0;
            lidar2.points(k,:) = {[]};
        end
        k = k+1;
    end
    
    human.t_rcv = (human.t_rcv - human.t_rcv(1))./1000;
    loc2.t_rcv = (loc2.t_rcv - loc2.t_rcv(1))./1000;
    clusters2.t_rcv = (clusters2.t_rcv - clusters2.t_rcv(1))./1000;
    lidar2.t_rcv = (lidar2.t_rcv - lidar2.t_rcv(1))./1000;
    
    clearvars k i j dist idx;

    % Roto-translate in absolute reference
    human_abs = [];
    clusters2_abs = [];
    lidar2_abs = [];
    for it = 1:length(loc2.t_rcv)
        % R and t
        angle = loc2.pose(it,3) + map_angle_offset;
        R = [cos(angle), -sin(angle); sin(angle), cos(angle)];
        t = loc2.pose(it,1:2);
        % Human
        human_abs(it,:) = (R*human.pos(it,:)')' + t;
        % Lidar clusters
        if not(isempty(clusters2.centroids{it}))
            clusters2_abs.centroids{it} = R*clusters2.centroids{it} + t';
        end
        for k = 1:size(clusters2.centroids{it,1},2)
            clusters2_abs.points{it,k} = (R*clusters2.points{it,k}')' + t;
        end
        % Lidar
        for k = 1:size(lidar2.points{it,1},1)
            lidar2_abs.points{it}(k,:) = (R*lidar2.points{it}(k,:)')' + t;
        end
    end
    clearvars it k R t;
    
    % Plot data
    if plotflag == true
        title(file, 'interpreter', 'latex');
        % human
        plot(human_abs(:,1), human_abs(:,2), 'r', 'LineWidth', 10);
        % robot loc
        plot(loc2.pose(:,1), loc2.pose(:,2), '*b', 'LineWidth', 1);
        % lidar clusters
        for it = 1:length(clusters2.t_rcv)
            if not(isempty(clusters2_abs.centroids{it}))
                plot(clusters2_abs.centroids{it}(1,:), clusters2_abs.centroids{it}(2,:), 'k', 'LineWidth', 1, 'Marker', 'o', 'MarkerFaceColor', 'k', 'linestyle', 'none');
                %text(clusters2_abs.centroids{it}(1,:)+0.05, clusters2_abs.centroids{it}(2,:)+0.05, lidar2.id{it}(1,:), cellstr(num2str(lidar2.id{it}')));
            end
        end
        % lidar clusters
        for it = 1:length(lidar2.t_rcv)
            if not(isempty(lidar2_abs.points{it}))
                plot(lidar2_abs.points{it}(:,1), lidar2_abs.points{it}(:,2), 'm', 'LineWidth', 1, 'Marker', '.', 'MarkerFaceColor', 'k', 'linestyle', 'none');
            end
        end
    end
    
    % Append data to Humans cells
    Humans{end+1,1} = [human_abs, (frame:frame+length(human.t_rcv)-1)', human.t_rcv]; % x-pos, y-pos, frame, time
    frame = frame+length(human.t_rcv);
end

Data.NumTraj = numel(Humans);
Data.Humans = Humans;
Data.SamplingTime = dt;

% Load estimated map
mapname = 'povo';
[Walls, x_lim, y_lim, map] = load_map_obstacles(mapname);

Data.Map.Walls = Walls;
Data.Map.xLim = x_lim;
Data.Map.yLim = y_lim;
Data.AxisLim = [16, 28.1, 7.5, 23];

% Plot
figure(3);
hold on, box on, grid on, axis equal;
axis(Data.AxisLim);

% Plot CAD map
%plot_obstacles('./maps/mappaRobodogC.txt', 'k');

% Plot estimated map
plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});


% Plot trajectories
for i = 1:numel(Humans)
    % clean human trajectories
    j = 1;
    while j <= length(Humans{i})
        if( discretize(Humans{i}(j,1), [24.88,24.91])==1 & discretize(Humans{i}(j,2), [15.58,15.61])==1 )
            Humans{i}(j,:) = [];
        else
            j = j+1;
        end
    end
    plot(Humans{i}(:,1), Humans{i}(:,2),'LineWidth',2);
    
end

% Save data
if saveflag == true
    save(['../povo_', num2str(date), '.mat'], 'Data');
end

disp('Done');
