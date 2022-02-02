% Compare PRM, RRT* and A* for human-like trajectory generation
% Placido Falqueto
% Alessandro Antonucci @AlexRookie
% University of Trento

clc;
close all;
clear all;

%==========================================================================

options.save = false; % save results
options.plot = true; % show plot
options.show = false; % show statistics

%==========================================================================

% Load dataset trajectories
load('povo_01_2022.mat');


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
subplot(1,3,3);
hold on, box on, grid on, axis equal;
axis(Data.AxisLim);
plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});
subplot(1,3,2);
hold on, box on, grid on, axis equal;
axis(Data.AxisLim);
plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});
subplot(1,3,1);
hold on, box on, grid on, axis equal;
axis(Data.AxisLim);
plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});

% Plot trajectories
for i = 1:numel(Humans)
    % clean human trajectories
    j = 1;
    while j <= length(Humans{i})
        if( discretize(Humans{i}(j,1), [24.88,24.91])==1 && discretize(Humans{i}(j,2), [15.58,15.61])==1 )
            Humans{i}(j,:) = [];
        else
            j = j+1;
        end
    end
%     plot(Humans{i}(:,1), Humans{i}(:,2),'LineWidth',2);
    subplot(1,3,1);
    plot(smooth(Humans{i}(:,1),10), smooth(Humans{i}(:,2),10),'LineWidth',2);
    subplot(1,3,2);
    plot(smooth(Humans{i}(:,1),10), smooth(Humans{i}(:,2),10),'LineWidth',2);
    subplot(1,3,3);
    plot(smooth(Humans{i}(:,1),10), smooth(Humans{i}(:,2),10),'LineWidth',2);

    generate_clothoid(map, [smooth(Humans{i}(:,1),10),smooth(Humans{i}(:,2),10)], Walls, [x_lim; y_lim], options);
    
%     pause(1);
    clf(figure(3))
    subplot(1,3,3);
    hold on, box on, grid on, axis equal;
    axis(Data.AxisLim);
    plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});
    subplot(1,3,2);
    hold on, box on, grid on, axis equal;
    axis(Data.AxisLim);
    plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});
    subplot(1,3,1);
    hold on, box on, grid on, axis equal;
    axis(Data.AxisLim);
    plot_map(Walls(:,2), 1, {[0.7,0.7,0.65],1});
end


disp('Done');